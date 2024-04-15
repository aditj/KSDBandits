
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from kdsd import KSD
import tqdm
N_arms = 10
true_p = np.random.uniform(0, 1, N_arms)
def sample_arm(arm,dist_parameters):
    return np.random.binomial(1, dist_parameters[arm], 1)

def get_reward(arm, dist_parameters):
    return sample_arm(arm, dist_parameters)

def get_best_arm(dist_parameters):
    return np.argmax(dist_parameters)

def get_regret(arm, dist_parameters):
    return dist_parameters[get_best_arm(dist_parameters)] - dist_parameters[arm]

def get_cumulative_regret(regret):
    return np.cumsum(regret)
### Score Function wrapper for Bernoulli distribution with parameter p
def score_function_bernoulli(x,p=0.5):
    return np.array(x/p - (1-x)/(1-p)).reshape(1,1)
def neg_function_bernoulli(x,l):
    return 1-x
def exp_hamming_kernel_1d(x, y):
    return np.exp(-np.abs(x-y))
def exp_hamming_kernel(x, y):
    """
    NOTE: The kernel matrix K is not symmetric, since in general
        K(x[i], y[j]) != K(x[j], y[i])
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    assert x.shape[1] == y.shape[1]  # d

    K = np.exp(-cdist(x, y, "Hamming"))

    return K
def ksd_est(kappa_vals_list):
    """
    Given a list of pre-computed kappa values, compute the U- and V-statistics
        estimates for KSD.

    Args:
        n: int, sample size.
        kappa_vals_list: list of array((n, n)), list of pre-computed kappa's.

    Returns: (all lists have same length as kappa_vals_list)
        ustats: list, U-statistics KSD estimate.
        vstats: list, V-statistics KSD estimate.
    """
    n = kappa_vals_list[0].shape[0]  # Sample size
    assert all(kappa_vals.shape == (n, n) for kappa_vals in kappa_vals_list)

    ustats = np.zeros(len(kappa_vals_list))  # U-stat
    vstats = np.zeros(len(kappa_vals_list))  # U-stat

    for i, kappa_vals in enumerate(kappa_vals_list):
        diag_vals = np.diag(np.diag(kappa_vals))  # (n, n) diagonal matrix
        ustats[i] = np.sum(kappa_vals - diag_vals) / (n * (n-1))  # U-stat
        vstats[i] = np.sum(kappa_vals) / (n**2)   # V-stat

    return ustats, vstats
def compute_ksd(q,samples_true,D=2):
    ksd = KSD(neg_fun= neg_function_bernoulli, score_fun=lambda x: score_function_bernoulli(q), kernel_fun=exp_hamming_kernel_1d)
    samples = np.zeros((D,D))
    
    for i in range(len(samples_true)):
        for j in range(len(samples_true)):
            samples[samples_true[i],samples_true[j]] += 1
    
    kappa_vals = ksd.compute_kappa(samples=samples)
    ksd_stats, _ = ksd_est([kappa_vals])
    return ksd_stats[0]


### Beta distribution parameters for modelling the distribution of each arm
prior_parameters = np.zeros((N_arms, 2))

### Number of times each arm is pulled
num_pulls = np.zeros(N_arms)
### Maximum Time Horizon 
T_Max = 2000

arms = np.arange(T_Max)
arms_thompson = np.arange(T_Max)
### Compute the regret
regret = np.zeros(T_Max)
### Data collected from each arm
data = [[] for _ in range(N_arms)]
total_data = [[] for _ in range(N_arms)]

### Sample each arm once
for arm in range(N_arms):
    for i in range(100):
        data[arm].append(int(sample_arm(arm, true_p)))
        num_pulls[arm] += 1
        total_data[arm] = data[arm].copy()

### Sample parameters from the prior distribution
dist_parameters = np.zeros(N_arms)

### Compute KSD of the sampled distribution with respect to the empirical data which is from the true distribution 
### for each arm
ksds = np.zeros((T_Max, N_arms))
N_Samples = 100
ksd = np.zeros(N_arms)
emp_mean = np.ones(N_arms)*0

### print ksd using true distribution
for arm in range(N_arms):
    print(sum(data[arm])/100,true_p[arm])
    print("KSD of arm ",arm," is ",compute_ksd(true_p[arm],data[arm]))


for t in tqdm.tqdm(range(T_Max)):
    ### Truncate data to last 1000 samples if the length of the data is greater than 1000
    for arm in range(N_arms):
        if len(total_data[arm]) > N_Samples:
            data[arm] = list(np.random.choice(total_data[arm], N_Samples))
        else:
            data[arm] = total_data[arm].copy()
    for arm in range(N_arms):
        dist_parameters[arm] = np.random.beta(1+prior_parameters[arm, 0], 1+prior_parameters[arm, 1], 1)


    for arm in range(N_arms):
        ksd[arm] = compute_ksd(dist_parameters[arm],data[arm])
        ksds[t, arm] = dist_parameters[arm]/ksd[arm]
    ### Sample arm with the highest KSD
    arm = np.argmax(emp_mean/ksd)
    thompson_arm = np.argmax(emp_mean)
    arms[t] = arm
    arms_thompson[t] = thompson_arm
    ### Pull the arm and collect data
    total_data[arm].append(int(sample_arm(arm, true_p)))
    ### Update the number of times the arm is pulled
    num_pulls[arm] += 1
    emp_mean[arm] = np.mean(total_data[arm])
    ### Update the posterior parameters
    prior_parameters[arm, 0] += total_data[arm][-1]
    prior_parameters[arm, 1] += 1 - total_data[arm][-1]

    regret[t] = get_regret(arm, true_p)

    cumulative_regret = get_cumulative_regret(regret[:t+1])
  

### Plot arm selection 
plt.plot(np.arange(T_Max), arms, label = "Arms using KSD")
plt.plot(np.arange(T_Max), arms_thompson, label = "Arms using Thompson Sampling")
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Arm")
plt.savefig("arm.png")
plt.close()

### Plot the cumulative regret
plt.plot(np.arange(T_Max), cumulative_regret)
plt.xlabel("Time Step")
plt.ylabel("Cumulative Regret")
plt.savefig("ksd.png")
plt.close()



### Plot the KSD of each arm
for arm in range(N_arms):
    plt.plot(np.arange(T_Max), ksds[:, arm], label = f"Arm {arm}")
plt.legend()
plt.title("Best Arm: "+str(get_best_arm(true_p)))
plt.xlabel("Time Step")
plt.ylabel("KSD")
plt.savefig("ksds.png")
