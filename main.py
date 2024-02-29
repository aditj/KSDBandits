
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from kdsd import KSD
import numpy as np

np.random.seed(123456)
DIV_MAX = 10000

def kl_divergence(p, q):
    if q == 0 and p == 0:
        return 0
    elif q == 0 and not p == 0:
        return DIV_MAX
    elif q == 1 and p == 1:
        return 0
    elif q == 1 and not p == 1:
        return DIV_MAX
    elif p == 0:
        return np.log(1/(1-q))
    elif p == 1:
        return np.log(1/q)
    return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

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

def compute_ksd(p, q):
    ksd = KSD(neg_fun= neg_function_bernoulli, score_fun=lambda x: score_function_bernoulli(p), kernel_fun=exp_hamming_kernel_1d)
    samples = np.random.binomial(1, q, size=(10,10))
    kappa_vals = ksd.compute_kappa(samples=samples)
    ksd_stats, _ = ksd_est([kappa_vals])
    return ksd_stats[0]

### Kernelized Stein Discrepancy
def ksd_confidence(t, emp_mean, num_pulls, precision = 1e-5, max_iter = 50):
    n = 0
    lower_bound = emp_mean
    upper_bound = 1
    while n < max_iter and upper_bound - lower_bound > precision:
        q = (lower_bound + upper_bound) / 2
        disc = compute_ksd(emp_mean, q)
        print(disc)
        if disc> (np.log(1 + t * np.log(t) ** 2)/num_pulls):
            upper_bound = q
        else:
            lower_bound = q
        n += 1
    return (lower_bound + upper_bound)/2.
def KSDUCB(arm_means, num_arms, total_steps):
    optimal_arm = np.argmax(arm_means)

    num_iterations = 1 # number of times we perform the same experiment

    regret = np.zeros([total_steps, num_iterations])

    for iter in range(num_iterations):
        emp_means = np.zeros(num_arms)
        num_pulls = np.zeros(num_arms)
        t = 0
        for step_count in range(0, total_steps):
            t += 1
            if step_count < num_arms:
                greedy_arm = step_count % num_arms
            else:
                # pick the best arm according to KL-UCB algorithm
                arm_confidence = np.zeros(num_arms)
                for idx in range(num_arms):
                    arm_confidence[idx] = ksd_confidence(t, emp_means[idx], num_pulls[idx])
                greedy_arm = np.argmax(arm_confidence)
                print(arm_confidence)

            # generate bernoulli reward from the picked greedy arm
            reward = np.random.binomial(1, arm_means[greedy_arm])
            num_pulls[greedy_arm] += 1
            regret[step_count, iter] += arm_means[optimal_arm] - arm_means[greedy_arm]
            emp_means[greedy_arm] += (reward - emp_means[greedy_arm])/num_pulls[greedy_arm]

    return regret 

def kl_confidence(t, emp_mean, num_pulls, precision = 1e-5, max_iter = 50):
    n = 0
    lower_bound = emp_mean
    upper_bound = 1
    while n < max_iter and upper_bound - lower_bound > precision:
        q = (lower_bound + upper_bound) / 2
        if kl_divergence(esmp_mean, q) > (np.log(1 + t * np.log(t) ** 2)/num_pulls):
            upper_bound = q
        else:
            lower_bound = q
        n += 1
    return (lower_bound + upper_bound)/2.


def KLUCB(arm_means, num_arms, total_steps):
  ### Choosing the optimal arm

  optimal_arm = np.argmax(arm_means)

  num_iterations = 10 # number of times we perform the same experiment
  
  regret = np.zeros([total_steps, num_iterations])
  
  for iter in range(num_iterations):
    emp_means = np.zeros(num_arms)
    num_pulls = np.zeros(num_arms)
    t = 0
    for step_count in range(0, total_steps):
        t += 1
        if step_count < num_arms:
            greedy_arm = step_count % num_arms
        else:
            # pick the best arm according to KL-UCB algorithm
            arm_confidence = np.zeros(num_arms)
            for idx in range(num_arms):
                arm_confidence[idx] = kl_confidence(t, emp_means[idx], num_pulls[idx])
            greedy_arm = np.argmax(arm_confidence)
        # generate bernoulli reward from the picked greedy arm
        reward = np.random.binomial(1, arm_means[greedy_arm])
        num_pulls[greedy_arm] += 1
        regret[step_count, iter] += arm_means[optimal_arm] - arm_means[greedy_arm]
        emp_means[greedy_arm] += (reward - emp_means[greedy_arm])/num_pulls[greedy_arm]

  return regret


algo_name = 'KSD-UCB'
num_arms_list = [2,5,7,10]
num_steps_list = [1000]

## Plot the regret curves
fig, axes = plt.subplots(2, len(num_arms_list)//2, figsize=(10, 10))

for idx, k in enumerate(num_arms_list):
  ### Generating distribution for each arm, we can choose any we like.
    ### We use dirichlet distribution in this case
    alpha = np.random.randint(1, k+1, k)
    arm_means = np.random.dirichlet(alpha, size = 1).squeeze(0)
    print(arm_means)
    legend = []

    for n in num_steps_list:
        regret = KSDUCB(arm_means, k, n)
        cum_regret = np.cumsum(regret, axis = 0)
        avg_regret = np.mean(cum_regret, axis = 1)
        axes[idx//2, idx%2].plot(np.arange(n), avg_regret)
    legend.append(f'n={n}')
    axes[idx//2, idx%2].set_title(f'Number of arms={k}')
    axes[idx//2, idx%2].legend(legend)
    axes[idx//2, idx%2].set_xlabel('Number of time steps', fontsize = 8)
    axes[idx//2, idx%2].set_ylabel('Cumulative Regret', fontsize = 8)
    for label in (axes[idx//2, idx%2].get_xticklabels() + axes[idx//2, idx%2].get_yticklabels()):
        label.set_fontsize(6)

    fig.suptitle(algo_name)
    plt.savefig("ksducb.png")