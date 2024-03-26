import numpy as np
import torch
### Bandits with Gaussian Rewards
N_arms = 10
N_experiments = 1
N_steps = 1000
true_means = np.random.normal(0, 1, N_arms)
from ksd import get_KSD
print(true_means)

def pull_arm(arm,true_means,sigma = 1):
    return np.random.normal(true_means[arm], sigma)

def update_estimates(arm, reward, N_pulls, estimates):
    estimates[arm] = (estimates[arm]*N_pulls[arm] + reward)/(N_pulls[arm]+1)
    return estimates

def epsilon_greedy(epsilon, estimates):
    if np.random.random() < epsilon:
        return np.random.randint(0, len(estimates))
    else:
        return np.argmax(estimates)

def thompson_sampling(estimates, N_pulls):
    samples = np.random.normal(estimates, 1/N_pulls)
    return np.argmax(samples)

def UCB(estimates, N_pulls, t):
    return np.argmax(estimates + np.sqrt(2*np.log(t)/N_pulls))

def KSD(estimates, N_pulls, samples, t):
    uncertainity = np.zeros(len(estimates))
    for arm in range(len(estimates)):
        kernel_type = 'rbf'
        h_method = 'dim'
        samples_arm = np.array(samples[arm]).reshape(-1,1)
        samples_arm = torch.from_numpy(samples_arm).float()
        mean = estimates[arm]
        ### Compute gradient of log likelihood of Gaussian with parameters mean and sigma at samples
        gradients = (samples_arm - mean)/1
        ### Compute KSD
        ksd_estimate = get_KSD(samples_arm, gradients, kernel_type, h_method)
        uncertainity[arm] = (ksd_estimate + estimates[arm])/N_pulls[arm]
    print(np.argmax(uncertainity))
    return np.argmax(uncertainity)
        ### Update estimates

### Evaluate Bernoulli vs Gaussian
#### Multiplying factor
### gradient computation
### regret doesn't tell much
### Exploratory metric





def run_experiment(N_arms, N_experiments, N_steps, true_means, epsilon = 0, run_thompson = False, run_UCB = False, run_KSD = False):
    regrets = np.zeros((N_experiments, N_steps))
    samples = [[] for _ in range(N_arms)]
    for i in range(N_experiments):
        estimates = np.zeros(N_arms)
        N_pulls = np.zeros(N_arms)
        ### Initial pulling
        for j in range(10):
            for arm in range(N_arms):
                reward = pull_arm(arm, true_means)
                samples[arm].append(reward)
                N_pulls[arm] += 1
                estimates = update_estimates(arm, reward, N_pulls, estimates)
        for arm in range(N_arms):
            reward = pull_arm(arm, true_means)
            samples[arm].append(reward)
            N_pulls[arm] += 1
            estimates = update_estimates(arm, reward, N_pulls, estimates)
        for t in range(N_steps):
            if run_thompson:
                arm = thompson_sampling(estimates, N_pulls)
            elif run_UCB:
                arm = UCB(estimates, N_pulls, t+1)
            elif run_KSD:
                arm = KSD(estimates, N_pulls, samples, t+1)
            else:
                arm = epsilon_greedy(epsilon, estimates)
            reward = pull_arm(arm, true_means)
            samples[arm].append(reward)
            regrets[i,t] = np.max(true_means) - true_means[arm]
            N_pulls[arm] += 1
            estimates = update_estimates(arm, reward, N_pulls, estimates)
    return np.cumsum(regrets, axis = 1)

regrets_KSD = run_experiment(N_arms, N_experiments, N_steps, true_means, run_KSD = True)

regrets_epsilon = run_experiment(N_arms, N_experiments, N_steps, true_means, epsilon = 0.1)
regrets_thompson = run_experiment(N_arms, N_experiments, N_steps, true_means, run_thompson = True)
regrets_UCB = run_experiment(N_arms, N_experiments, N_steps, true_means, run_UCB = True)

import matplotlib.pyplot as plt
plt.plot(np.mean(regrets_epsilon, axis = 0), label = 'epsilon-greedy')
plt.plot(np.mean(regrets_thompson, axis = 0), label = 'thompson sampling')
plt.plot(np.mean(regrets_UCB, axis = 0), label = 'UCB')
plt.plot(np.mean(regrets_KSD, axis = 0), label = 'KSD')
plt.legend()
plt.savefig('gaussian_bandits.png')


#

