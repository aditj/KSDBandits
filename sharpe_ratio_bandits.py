import numpy as np
import matplotlib.pyplot as plt
import tqdm 
from ksd import get_KSD
import torch
# from scipy.stats import norm

def pull_arm(arm,mean_mixtures,variance_mixtures,weight_mixtures):
    component = np.random.choice(np.arange(N_components),p=weight_mixtures[arm])
    return np.random.normal(mean_mixtures[arm,component],variance_mixtures[arm,component])
def compute_sharpe_ratio_mixture(mean_mixtures,variance_mixtures):
    sharpe_ratio = np.zeros(mean_mixtures.shape[0])
    for i in range(mean_mixtures.shape[0]):
        sharpe_ratio[i] = np.max(mean_mixtures[i]/variance_mixtures[i])
    return sharpe_ratio
def gradient_gmm_parameters(weights, means, std, x):
    n_components = len(weights)
    n_samples = len(x)
    gradient = torch.zeros((n_samples,n_components))
    dist = lambda mean,std: torch.distributions.normal.Normal(mean, std)
    norm = lambda x,mean,std: torch.exp(dist(mean,std).log_prob(x))
    ## print types of weights, means, and mean_variance and norm.pdf
    for i in range(n_components):
        for j in range(n_samples):
            gradient[j,i] = weights[i]*norm(x[j], means[i], torch.sqrt(std[i]))*torch.tensor(x[j]-means[i])/(torch.sum(weights*norm(x[j], means, torch.sqrt(std)))*std[i]**2)
    return gradient

### Algorithms
### Basic EM algorithm for Gaussian Mixtures
def em_algorithm(samples,N_components = 2, N_iter = 100):
    weights = np.random.dirichlet(np.ones(N_components))
    means = np.random.normal(0,1,N_components)
    variance = np.random.uniform(0.1,1,N_components)
    print(samples)
    samples = np.array(samples)
    N_samples = samples.shape[0]
    for i in range(N_iter):
        ### E Step
        responsibilities = np.zeros((N_samples,N_components))
        for j in range(N_samples):
            for k in range(N_components):
                responsibilities[j,k] = weights[k]*np.exp(-0.5*(samples[j]-means[k])**2/variance[k])/np.sqrt(2*np.pi*variance[k])
            responsibilities[j] /= np.sum(responsibilities[j])
        ### M Step
        N_k = np.sum(responsibilities,axis=0)
        weights = N_k/N_samples
        means = np.sum(responsibilities*samples.reshape(-1,1),axis=0)/N_k
        variance = np.sum(responsibilities*(samples.reshape(-1,1)-means)**2,axis=0)/N_k
    return weights, means, variance
### Thompson Sampling
def thompson_sampling(samples,N_components = 2, N_iter = 100,N_arms=10):
    reward_samples = np.zeros(N_arms)
    for i in range(N_arms):
        weights, means, variance = em_algorithm(samples[i],N_components = N_components, N_iter = N_iter)
        weight_sample = np.random.dirichlet(weights)
        mean_arms = np.random.normal(means,variance)
        mean_arm = np.random.normal(means,variance)
        reward_samples[i] = np.max(mean_arms/variance)
    return np.argmax(reward_samples)

def ksd_index(armSamples,lambda_):
    """
    :numSucs: samples of the arm
    """
    N_iter = 100
    device = torch.device('mps')
    kernel_type = 'rbf' 
    h_method = 'dim'
    max_value = 0
    #### convert to tensor of type float32
    ### convert to float 
    weights, means, variance = em_algorithm(armSamples,N_components = N_components, N_iter = N_iter)
    print(armSamples,weights,means,variance)
    sharpe_ratio_estimate = np.max(np.divide(means,variance))

    weights = torch.from_numpy(weights).float().reshape(-1,1).to(device)
    means = torch.from_numpy(means).float().reshape(-1,1).to(device)
    variance = torch.from_numpy(variance).float().reshape(-1,1).to(device)
    armSamples = torch.from_numpy(np.array(armSamples,dtype=np.float32)).reshape(-1,1).to(device)

    gradients = gradient_gmm_parameters(weights,means,variance,armSamples).to(device)
   # gradients = torch.from_numpy(gradients).float().reshape(-1,1)
    ksd_value = get_KSD(armSamples, gradients, kernel_type, h_method).cpu().numpy()

    return sharpe_ratio_estimate + lambda_*ksd_value

def ksd_ucb(samples,lambda_):
    """
    :[i]: arm i
    :samples[i]: samples of arm i
    :lambda_: lambda value for KSD-UCB
    """
    ### Algorithm: 
   
    ksd_indices = np.zeros(len(samples))
    for i in range(len(samples)):
        armSamples = samples[i]

        ### Compute KSD
        ksd_indices[i] = ksd_index(armSamples,lambda_)
       
        
    return np.argmax(ksd_indices)


### Gaussian Mixture Bandits
N_arms = 2
N_components = 2

mean_mixtures = np.random.normal(0,1,(N_arms,N_components))
variance_mixtures = np.random.uniform(0.1,1,(N_arms,N_components))
weights = np.random.dirichlet(np.ones(N_components),N_arms)

sharpe_ratios = compute_sharpe_ratio_mixture(mean_mixtures,variance_mixtures).reshape(-1,1)
assert sharpe_ratios.shape == (N_arms,1)


optimal_arm = np.argmax(sharpe_ratios)
print(sharpe_ratios)
print(optimal_arm, " is the optimal arm")

###
T = 1000
N_methods = 2
regret = np.zeros((N_methods,T))
total_data =[ [[] for i in range(N_arms)] for j in range(N_methods)]
sample_max = 100
for _ in range(N_arms):
    for j in range(10):
    
        total_data[0][_].append(pull_arm(_,mean_mixtures,variance_mixtures,weights))
        total_data[1][_].append(pull_arm(_,mean_mixtures,variance_mixtures,weights))
for t in tqdm.tqdm(range(T)):
    ### pull arm 10 times for each method
    
    arm = thompson_sampling(total_data[0],N_components = N_components, N_iter = 100,N_arms=N_arms)
    reward = pull_arm(arm,mean_mixtures,variance_mixtures,weights)
    regret[0,t] = np.max(sharpe_ratios) - sharpe_ratios[arm]
    total_data[0][arm].append(reward)
    for arm_idx in range(N_arms):
        if len(total_data[0][arm_idx])>sample_max:
            total_data[0][arm_idx] = list(np.random.choice(total_data[0][arm_idx],sample_max,replace=False))

    arm = ksd_ucb(total_data[1],lambda_=0.1)
    reward = pull_arm(arm,mean_mixtures,variance_mixtures,weights)
    regret[1,t] = np.max(sharpe_ratios) - sharpe_ratios[arm]
    total_data[1][arm].append(reward)
    for arm_idx in range(N_arms):
        if len(total_data[1][arm_idx])>sample_max:
            total_data[1][arm_idx] = list(np.random.choice(total_data[1][arm_idx],sample_max,replace=False))



cumulative_regret = np.cumsum(regret,axis=1)
plt.figure()
plt.plot(np.arange(T), cumulative_regret[0],label="Thompson Sampling")
plt.plot(np.arange(T), cumulative_regret[1],label="KSD-UCB")
plt.xlabel("Time Step")
plt.ylabel("Cumulative Regret")
plt.savefig("plots/cumulative_regret_sharperatio.png")
