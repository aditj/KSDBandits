import numpy as np
import gym
import heapq
import argparse
import json
import os
from pendulum_gym import PendulumEnv
from cartpole_continuous import ContinuousCartPoleEnv
import torch
import time
from CEM_without import CEM
import scipy.stats as stats
from NB_dx_tf import neural_bays_dx_tf

from tf_models.constructor import construct_shallow_model, construct_shallow_cost_model, construct_model, construct_cost_model

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='Pendulum-v0', metavar='ENV',
                        help='env :[Pendulum-v0, CartPole-v0,CartPole-continuous]')
    parser.add_argument('--with-reward', type=bool, default=True, metavar='NS',
                        help='predict with true rewards or not')
    parser.add_argument('--sigma', type=float, default=1e1, metavar='T', help='var for betas')
    parser.add_argument('--sigma_n', type=float, default=1e-3, metavar='T', help='var for noise')
    parser.add_argument('--num-elites', type=int, default=5, metavar='NS', help='number of choosing best params')
    parser.add_argument('--num-trajs', type=int, default=100, metavar='NS',
                        help='number of sampling from params distribution')
    parser.add_argument('--alpha', type=float, default=0., metavar='T',
                        help='Controls how much of the previous mean and variance is used for the next iteration.')
    parser.add_argument('--plan-hor', type=int, default=30, metavar='NS', help='number of choosing best params')
    parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
    parser.add_argument('--max-iters', type=int, default=5, metavar='NS', help='iteration of cem')
    parser.add_argument('--epsilon', type=float, default=0.001, metavar='NS', help='threshold for cem iteration')
    parser.add_argument('--hidden-dim-dx', type=int, default=200, metavar='NS')

    parser.add_argument('--hidden-dim-cost', type=int, default=200, metavar='NS')

    parser.add_argument('--training-iter-dx', type=int, default=100, metavar='NS')
    parser.add_argument('--training-iter-cost', type=int, default=150, metavar='NS')
    parser.add_argument('--var', type=float, default=3.0, metavar='T', help='var')
    parser.add_argument('--path', type=str, default=None, metavar='NS', help='output_path')
    parser.add_argument('--predict_with_bias', type=bool, default = True, metavar='NS',
                        help='predict y with bias')

    args = parser.parse_args()

    # Set a random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("current dir:", os.getcwd())
    if 'CartPole-continuous' in args.env:
        env = ContinuousCartPoleEnv()
    elif 'Pendulum-v0' in args.env:
        env = PendulumEnv()
    else:
        env = gym.make(args.env)

    print('env', env)

    slb = env.observation_space.low
    sub = env.observation_space.high
    alb = env.action_space.low
    aub = env.action_space.high

    obs_shape = env.observation_space.shape[0]
    action_shape = len([env.action_space.sample()])

    dx_model = construct_shallow_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)
    if not args.with_reward:
        cost_model = construct_shallow_cost_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)

    my_dx = neural_bays_dx_tf(args, dx_model, "dx", obs_shape, sigma_n2=args.sigma_n**2,sigma2=args.sigma**2)
    if not args.with_reward:
        my_cost = neural_bays_dx_tf(args, cost_model, "cost", 1, sigma_n2 = args.sigma_n**2,sigma2 = args.sigma**2)

    posterior_mu, posterior_sigma = [], []
    t1 = time.time()
    #get the cumulative data
    list_model_order = []
    cum_rewards = []
    ksd_total = []
    #the number of episodes are 15
    num_episode = 50
    for episode in range(num_episode):
        if args.with_reward:
            from CEM_with import CEM
            cem = CEM(env, args, my_dx, num_elites=args.num_elites, num_trajs=args.num_trajs, alpha=args.alpha)
        else:
            from CEM_without import CEM
            cem = CEM(env, args, my_dx, my_cost, num_elites=args.num_elites, num_trajs=args.num_trajs, alpha=args.alpha)
        state = torch.tensor(env.reset())
        if 'Pendulum-v0' in args.env:
            state = state.squeeze()
        time_step = 0
        done = False
        my_dx.sample()
        if not args.with_reward:
            my_cost.sample()
        num_steps = 200
        cum_reward = 0

        for _ in range(num_steps):
            if episode == 0:
                best_action = env.action_space.sample()
            else:
                #using mpc method : choose the best action
                best_action = cem.hori_planning(state)

            if 'Pendulum-v0' in args.env:
                best_action = np.array([best_action])

            #get new data here in this step
            new_state, r, done, _ = env.step(best_action)

            r = torch.tensor(r)

            new_state = torch.tensor(new_state)
            if 'Pendulum-v0' in args.env:
                new_state = new_state.squeeze()
                best_action = best_action.squeeze(0)
                r = r.squeeze(0)

            #here for every time step, we are adding new data in my_dx : so for 200 timesteps we are getting 200 data
            xu = torch.cat((state.double(), torch.tensor(best_action).double()))

            
            
            shape = my_dx.add_data(new_x=xu, new_y=new_state - state)
                        
            if not args.with_reward:
                my_cost.add_data(new_x=xu, new_y=r)
            
            cum_reward += r

            state = new_state

        print(episode, ': cumulative rewards', cum_reward.item())

        cum_rewards.append([episode, cum_reward.tolist()])

        #get the shape of the data after every episode
        # list_model_order.append(my_dx.get_shape())

        #training the model on the entire data
        my_dx.train(args.training_iter_dx)

        #updating the rewards model
        if not args.with_reward:
            my_cost.train(args.training_iter_cost)
            my_cost.update_bays_reg()
        
        np.savetxt(args.path, cum_rewards)
    

    print(cum_rewards)

    #save 

