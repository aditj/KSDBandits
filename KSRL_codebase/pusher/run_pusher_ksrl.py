import numpy as np
import gym
import heapq
import argparse
import json
import time
import os
from pendulum_gym import PendulumEnv
from cartpole_continuous import ContinuousCartPoleEnv
from pusher import PusherEnv
import torch
import scipy.stats as stats
from tf_models.constructor import construct_model, construct_cost_model
# from NB_dx_tf import  neural_bays_dx_tf
from NB_dx_tf_new import  neural_bays_dx_tf
from CEM_without import CEM
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

t1 = time.time()

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--with-reward', type=bool, default=True, metavar='NS',
                        help='predict with true rewards or not')

    parser.add_argument('--predict_with_bias', type=bool, default = True, metavar='NS',
                        help='predict y with bias')
    parser.add_argument('--sigma', type=float, default=1e-03, metavar='T', help='var for betas')
    parser.add_argument('--sigma_n', type=float, default=1e-04, metavar='T', help='var for noise')
    parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
    parser.add_argument('--var', type=float, default=1.0, metavar='T', help='var')
    parser.add_argument('--num-trajs', type=int, default=500, metavar='NS',
                        help='number of sampling from params distribution')
    parser.add_argument('--num-elites', type=int, default=50, metavar='NS', help='number of choosing best params')

    parser.add_argument('--hidden-dim-dx', type=int, default = 200, metavar='NS')
    parser.add_argument('--training-iter-dx', type=int, default=50, metavar='NS')
    parser.add_argument('--hidden-dim-cost', type=int, default = 200, metavar='NS')
    parser.add_argument('--training-iter-cost', type=int, default=50, metavar='NS')

    parser.add_argument('--alpha', type=float, default=0.1, metavar='T',
                        help='Controls how much of the previous mean and variance is used for the next iteration.')
    parser.add_argument('--env', default='Pusher', metavar='ENV', help='env :[Pendulum-v0, CartPole-v0,CartPole-continuous]')
    parser.add_argument('--num-iters', type=int, default=100, metavar='NS', help='number of iterating the distribution params')
    parser.add_argument('--plan-hor', type=int, default=25, metavar='NS', help='number of choosing best params')
    parser.add_argument('--max-iters', type=int, default=5, metavar='NS', help='iteration of cem')
    parser.add_argument('--path', type=str, default=None, metavar='NS', help='output_path')

    
    args = parser.parse_args()
    
    # Set a random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("current dir:", os.getcwd())
    if 'CartPole-continuous' in args.env:
        env = ContinuousCartPoleEnv()
    elif 'Pendulum-v0' in args.env:
        env = PendulumEnv()
    elif "Pusher" in args.env:
        env = PusherEnv()
    else:
        env = gym.make(args.env)
    
    

    print('env', env)

    slb = env.observation_space.low
    sub = env.observation_space.high
    alb = env.action_space.low
    aub = env.action_space.high
    obs_shape = env.observation_space.shape[0]
    action_shape = len(env.action_space.sample())
    model = construct_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)
    
    print (args.with_reward)
    
    if not args.with_reward:
        cost_model = construct_cost_model(obs_dim=obs_shape, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)


    my_dx = neural_bays_dx_tf(args, model, "dx", obs_shape, sigma2 = args.sigma**2, sigma_n2 = args.sigma_n**2)

    if not args.with_reward:
        my_cost = neural_bays_dx_tf(args, cost_model, "cost", 1, sigma2 = args.sigma**2, sigma_n2 = args.sigma_n**2)

    t1 = time.time()
    cum_rewards = []
    list_model_order = []
    ksd_total = []
    num_episode = 100
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
        num_steps = 150
        cum_reward = 0
        
        for _ in range(num_steps):
            if episode == 0:
                best_action = env.action_space.sample()
            else:
                best_action = cem.hori_planning(state)

            if 'Pendulum-v0' in args.env:
                best_action = np.array([best_action])

            new_state, r, done, _ = env.step(best_action)

            r = torch.tensor(r)

            new_state = torch.tensor(new_state)
            if 'Pendulum-v0' in args.env:
                new_state = new_state.squeeze()
                best_action = best_action.squeeze(0)
                r = r.squeeze(0)

            xu = torch.cat((state.double(), torch.tensor(best_action).double()))

            
            my_dx.add_data(new_x=xu, new_y=new_state - state, new_r = r)
            
            if not args.with_reward:
                my_cost.add_data(new_x=xu, new_y=r, new_r = r)
            
            #add rewards
            cum_reward += r

            state = new_state

        print(episode, ': cumulative rewards', cum_reward.item())

        cum_rewards.append([episode, cum_reward.tolist()])
        
        #save
        list_model_order.append(my_dx.get_shape())

        
        
        my_dx.train(args.training_iter_dx)
        my_dx.update_bays_reg()
        
        if not args.with_reward:
            my_cost.train(args.training_iter_cost)
            
        ksd_val = my_dx.thin_data_cole('ksd', thin_sampl)

        np.savetxt(args.path, cum_rewards)
        
    # print(cum_rewards)
    