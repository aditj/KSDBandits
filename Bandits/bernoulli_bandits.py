#### Bandits with Bernoulli Rewards
import numpy as np
import matplotlib.pyplot as plt

### Bandits with Bernoulli Rewards
N_arms = 10
N_experiments = 1000
N_steps = 1000
true_means = np.random.random(N_arms)
print(true_means)

def pull_arm(arm,true_means):
    return np.random.random() < true_means[arm]

def update_estimates(arm, reward, N_pulls, estimates):
    estimates[arm] = (estimates[arm]*N_pulls[arm] + reward)/(N_pulls[arm]+1)
    return estimates

def epsilon_greedy(epsilon, estimates):
    if np.random.random() < epsilon:
        return np.random.randint(0, len(estimates))
    else:
        return np.argmax(estimates)

def thompson_sampling(estimates, N_pulls):
    samples = np.random.beta(estimates*N_pulls+1, (1-estimates)*N_pulls+1)
    return np.argmax(samples)

def UCB(estimates, N_pulls, t):
    rewards = estimates + np.sqrt(2*np.log(t)/N_pulls)
    
    return np.argmax(rewards)

def run_experiment(N_arms, N_experiments, N_steps, true_means, epsilon = 0, run_thompson = False, run_UCB = False,reward_type = 'bernoulli'):
    regrets = np.zeros((N_experiments, N_steps))
    for i in range(N_experiments):
        estimates = np.zeros(N_arms)
        N_pulls = np.zeros(N_arms)
        ### Initial pulling
        for arm in range(N_arms):
            reward = pull_arm(arm, true_means)
            N_pulls[arm] += 1
            estimates = update_estimates(arm, reward, N_pulls, estimates)
        for t in range(N_steps):
            if run_thompson:
                arm = thompson_sampling(estimates, N_pulls)
            elif run_UCB:
                arm = UCB(estimates, N_pulls, t+1)
            else:
                arm = epsilon_greedy(epsilon, estimates)
            reward = pull_arm(arm, true_means)
            regrets[i,t] = np.max(true_means) - true_means[arm]
            N_pulls[arm] += 1
            estimates = update_estimates(arm, reward, N_pulls, estimates)
    return np.cumsum(regrets, axis = 1)

REWARD_TYPE = 'bernoulli'
regrets_epsilon = run_experiment(N_arms, N_experiments, N_steps, true_means, epsilon = 0.1,reward_type = REWARD_TYPE)
regrets_thompson = run_experiment(N_arms, N_experiments, N_steps, true_means, run_thompson = True,reward_type = REWARD_TYPE)
regrets_UCB = run_experiment(N_arms, N_experiments, N_steps, true_means, run_UCB = True,reward_type = REWARD_TYPE)

plt.plot(np.mean(regrets_epsilon, axis = 0), label = 'epsilon-greedy')
plt.plot(np.mean(regrets_thompson, axis = 0), label = 'thompson sampling')
plt.plot(np.mean(regrets_UCB, axis = 0), label = 'UCB')
plt.legend()
plt.savefig(f'plots/{REWARD_TYPE}_bandits_regret_{N_arms}_{N_experiments}_{N_steps}.png')

