import numpy as np
import matplotlib.pyplot as plt
import tqdm
from utils import sample_arm
#### True Arm Means ####
# True means of the arms
N_arms = 4 # Number of arms
N_components = 2 # Number of components in the GMM

true_means = np.array([[1.0,1.8],
[0.2,2.4],
[0.5,1.5],
[0.1,2.7]]) # True means of the arms

variance_arm = 0.1  
true_variance = np.array([variance_arm,variance_arm]) # True variance of the arm mixture components

true_weights = np.array([0.5,0.5]) # True weights of the arms

### Calculate the best arm ###
def calculate_best_arm_basic(samples, t, N_arms):
    mean_rewards = np.zeros(N_arms)
    for arm_idx in range(N_arms):
        mean_rewards[arm_idx] = np.mean(samples[arm_idx])
    return mean_rewards

def estimate_mean(samples,N_components = 2):
    mean_estimates = np.zeros(N_components)
    sorted_samples = np.sort(samples)
    diff = np.diff(sorted_samples)

    split_idx = np.argmax(diff) 
    mean_estimates[0] = np.mean(sorted_samples[0:split_idx])
    mean_estimates[1] = np.mean(sorted_samples[split_idx:])
    return mean_estimates

def calculate_best_arm(samples,  N_arms):
    mean_rewards = np.zeros(N_arms)
    for arm_idx in range(N_arms):
        mean_estimates = estimate_mean(samples[arm_idx])
        mean_rewards[arm_idx] = np.max(mean_estimates)
    return mean_rewards
### benchmark epsilon-greedy basic (assuming single gaussian) with epsilon-greedy with GMM of 2 components

N_mc = 100 # Number of Monte Carlo simulations
T = 10000 # Number of rounds

epsilon = 0.1 # Epsilon for epsilon-greedy
epsilon_gmm = 0.1 # Epsilon for epsilon-greedy with GMM
# Initialize variables to store the rewards
rewards_basic = np.zeros((N_mc,T))
rewards_gmm = np.zeros((N_mc,T))
samples_basic = [[[] for i in range(N_arms)] for _ in range(N_mc)]
samples = [[[] for i in range(N_arms)] for _ in range(N_mc)]

RUN_EXP = 1
if RUN_EXP:
    for mc in tqdm.tqdm(range(N_mc)):
        for t in range(T):
            # Epsilon-greedy basic
            if np.random.rand() < epsilon:
                # Explore
                arm_idx = np.random.choice(N_arms)
            else:
                # Exploit
                arm_idx = np.argmax(calculate_best_arm_basic(samples_basic[mc], t, N_arms))
            samples_basic[mc][arm_idx].append(sample_arm(true_means, true_variance, true_weights, arm_idx))                
            rewards_basic[mc,t] = np.max(true_means[arm_idx])
            #epsilon *= 0.99
            # Epsilon-greedy with GMM
            if t<100:
                arm_idx = t%N_arms
            else:
                if np.random.rand() < epsilon_gmm:
                    # Explore
                    arm_idx = np.random.choice(N_arms)
                else:
                    # Exploit

                    arm_idx = np.argmax(calculate_best_arm(samples[mc], N_arms))
                #epsilon_gmm *= 0.99
            samples[mc][arm_idx].append(sample_arm(true_means, true_variance, true_weights, arm_idx))
            rewards_gmm[mc,t] = np.max(true_means[arm_idx])

    np.save('parameters/rewards_basic',rewards_basic)
    np.save('parameters/rewards_gmm',rewards_gmm)


rewards_basic = np.load('parameters/rewards_basic.npy')
rewards_gmm = np.load('parameters/rewards_gmm.npy')

### pseudo-regret calculation
best_mean = np.max(true_means)

regret_basic = np.zeros((N_mc,T))
regret_gmm = np.zeros((N_mc,T))
regret_basic = np.cumsum(best_mean - rewards_basic, axis=1)
regret_gmm = np.cumsum(best_mean - rewards_gmm, axis=1)


### Plotting with error bars
plt.figure()
plt.plot(np.mean(regret_basic,axis=0),label='Epsilon-Greedy Basic')
plt.plot(np.mean(regret_gmm,axis=0),label='Epsilon-Greedy with GMM')

plt.fill_between(np.arange(T),np.mean(regret_basic,axis=0) - 2*np.std(regret_basic,axis=0)/np.sqrt(N_mc),np.mean(regret_basic,axis=0) + 2*np.std(regret_basic,axis=0)/np.sqrt(N_mc),alpha=0.2)
plt.fill_between(np.arange(T),np.mean(regret_gmm,axis=0) - 2*np.std(regret_gmm,axis=0)/np.sqrt(N_mc),np.mean(regret_gmm,axis=0) + 2*np.std(regret_gmm,axis=0)/np.sqrt(N_mc),alpha=0.2)
plt.xlabel('Rounds')
plt.title(f"Pseudo Regret of $\epsilon$-greedy for GMM with variance = {variance_arm} and {N_arms} arms")
plt.ylabel('Pseudo-Regret')
plt.legend()
plt.savefig(f'plots/pseudo_regret_epsilongreedy_{variance_arm}_{N_arms}.png')