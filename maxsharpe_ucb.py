import numpy as np
import matplotlib.pyplot as plt
import tqdm

N_arms = 2 # Number of arms
N_components = 2 # Number of components in the GMM

true_means = np.array([[1.0,1.8],[0.1,2.7]]) # True means of the arms
 
true_variance = np.array([0.1,0.1]) # True variance of the arms

true_weights = np.array([0.5,0.5]) # True weights of the arms

### Generate Data ###
def sample_arm(true_means, true_variance, true_weights, arm_idx):
    """
    Sample from the arm with index arm_idx.
    """
    N_arms = len(true_means)
    N_components = len(true_weights)
    # Sample the component
    component_idx = np.random.choice(N_components, p=true_weights)
    # Sample the arm
    sample = np.random.normal(true_means[arm_idx,component_idx],np.sqrt(true_variance[arm_idx]))
    return sample


def estimate_max_sharpe_ratio(samples,N_components = 2):
    mean_estimates = np.zeros(N_components)
    
    sorted_samples = np.sort(samples)
    diff = np.diff(sorted_samples)

    split_idx = np.argmax(diff) 
    mean_estimates[0] = np.mean(sorted_samples[0:split_idx])
    mean_estimates[1] = np.mean(sorted_samples[split_idx:])
    min_samples = min(split_idx,len(samples)-split_idx)
    confidence = np.sqrt(np.log(len(samples))/min_samples)
    return max(mean_estimates),confidence
    

### benchmark ucb basic (assuming single gaussian) with ucb with GMM of 2 components

N_mc = 10 # Number of Monte Carlo simulations

T = 10000 # Number of rounds

# Initialize variables to store the rewards
rewards_basic = np.zeros((N_mc,T))
rewards_gmm = np.zeros((N_mc,T))

# Run the simulatins
RUN_UCB = 1
if RUN_UCB:
    for i in tqdm.tqdm(range(N_mc)):
        
        # Initialize the samples
        samples_basic = [[] for _ in range(N_arms)]
        samples_gmm = [[] for _ in range(N_arms)]
        # Initialize the number of times each arm is pulled
        N_pulls = np.zeros(N_arms)
        # Initialize the rewards
        # Pull each arm once
        for _ in range(100):
            for arm_idx in range(N_arms):
                sample = sample_arm(true_means, true_variance, true_weights, arm_idx)
                samples_basic[arm_idx].append(sample)
                samples_gmm[arm_idx].append(sample)
                N_pulls[arm_idx] += 1
        # Pull each arm T times
        for t in range(100*N_arms//2,T):
            ### Classical UCB 
            # Calculate the mean rewards
            mean_rewards = np.zeros(N_arms)
            for arm_idx in range(N_arms):
                mean_rewards[arm_idx] = np.mean(samples_basic[arm_idx])
            # Calculate the upper confidence bounds
            UCB = mean_rewards + np.sqrt(2*np.log(t)/N_pulls)
            # Choose the arm with the highest UCB
            arm_idx = np.argmax(UCB)
            # Sample from the chosen arm
            sample = sample_arm(true_means, true_variance, true_weights, arm_idx)
            # Update the samples
            samples_basic[arm_idx].append(sample)
            
            # Update the number of pulls
            N_pulls[arm_idx] += 1
            # Store the rewards
            rewards_basic[i,t] = np.max(true_means[arm_idx])

            ### UCB with GMM
            mean_rewards = np.zeros(N_arms)

            for arm_idx in range(N_arms):
                mean_estimates,confidence = estimate_max_sharpe_ratio(samples_gmm[arm_idx])
                mean_rewards[arm_idx] = mean_estimates
            # Calculate the upper confidence bounds
            UCB = mean_rewards + confidence
            # Choose the arm with the highest UCB
            arm_idx = np.argmax(UCB)
            # Sample from the chosen arm
            sample = sample_arm(true_means, true_variance, true_weights, arm_idx)
            # Update the samples
            samples_gmm[arm_idx].append(sample)
            # Update the number of pulls
            N_pulls[arm_idx] += 1
            # Store the rewards
            rewards_gmm[i,t] = np.max(true_means[arm_idx])

    np.save('parameters/rewards_basic_ucb',rewards_basic)
    np.save('parameters/rewards_gmm_ucb',rewards_gmm)


rewards_basic = np.load('parameters/rewards_basic_ucb.npy')
rewards_gmm = np.load('parameters/rewards_gmm_ucb.npy')

### pseudo-regret calculation
best_mean = np.max(true_means)

regret_basic = np.zeros((N_mc,T))
regret_gmm = np.zeros((N_mc,T))
regret_basic = np.cumsum(best_mean - rewards_basic, axis=1)
regret_gmm = np.cumsum(best_mean - rewards_gmm, axis=1)


# Plot the results
plt.figure()
plt.plot(np.mean(regret_basic,axis=0),label='UCB Basic')
plt.plot(np.mean(regret_gmm,axis=0),label='UCB with GMM')

# plt.fill_between(np.arange(T),np.mean(regret_basic,axis=0) - 2*np.std(regret_basic,axis=0)/np.sqrt(N_mc),np.mean(regret_basic,axis=0) + 2*np.std(regret_basic,axis=0)/np.sqrt(N_mc),alpha=0.2)
# plt.fill_between(np.arange(T),np.mean(regret_gmm,axis=0) - 2*np.std(regret_gmm,axis=0)/np.sqrt(N_mc),np.mean(regret_gmm,axis=0) + 2*np.std(regret_gmm,axis=0)/np.sqrt(N_mc),alpha=0.2)
plt.legend()
plt.xlabel('Rounds')
plt.ylabel('Pseudo-regret')
plt.title('Pseudo-regret of UCB with GMM')
plt.savefig('plots/ucb_gmm.png')

