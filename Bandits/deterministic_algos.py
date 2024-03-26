
# coding: utf-8

# In[ ]:

import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import time
import pickle
import torch

from ksd import get_KSD

"""
Deterministic algorithms include
"ucb", "ucb_tuned", "gpucb", "gpucb_tuned", "kl_ucb", "bayes_ucb", "kg", "kg_star", "bmle"
"""

#-----inputs-----
path = "plots" # directory to save results
means = [0.41, 0.32]#, 0.66, 0.43]#, 0.58] # 0.65, 0.48, 0.67, 0.59, 0.63] # means of Gaussian distribution
excelID = 16 # excelID is used to dfferentiate which set of probs to be tested
seed = 46 # seed number for reproducibility and ensure the data prepared in advance is the same for all parallel programs
numExps = 100 # total number of trials
T = int(1e3) # time horizon T

#-----funcs-----
def compute_bias(numSucs, numPulls, t, beta_t, sigma):
    numSucs, numPulls = np.array(numSucs), np.array(numPulls)
    numArms = len(numSucs)
    p_t = numSucs / numPulls
    term = np.sqrt((numArms + 2)*np.log(t) / numPulls)
    u_t = p_t + term
    l_t = p_t - term
    u_max = max(u_t)
    l_min = min(l_t)
    
    temp = [(u_t[i], i) for i in range(numArms)]
    temp.sort()
    cs = np.zeros(numArms)
    for i in range(numArms):
        ind = temp[i][1]
        if i == numArms-1:
            cs[ind] = temp[i-1][0]
        else:
            cs[ind] = temp[-1][0]
            
    delta_t = max(np.maximum(0, l_t-cs))
    
    c_alpha_t = 256.0*sigma**2/delta_t
    
    if beta_t > c_alpha_t * np.log(t):
        return c_alpha_t * np.log(t)
    else:
        return beta_t * np.log(t)

def bmle(numSucs, numPulls, t, beta_t, sigma):
    """
    :Implementation of Algorithm 1 in the submitted paper
    numSucs[i]: S_i(t) num of successes in pulling arm i
    numPulls[i]: N_i(t) total num of pulls of arm i
    beta_t: beta(t) the bias term at the t-th round of decision
    Returns: the index of the arm with the largest index, Time complexity O(nlog(n)), n = num of arms
    """
    bias = compute_bias(numSucs, numPulls, t, beta_t, sigma)
    rst, indVal = 0, float('-inf')
    for i in range(len(numSucs)):
        curtVal = float(numSucs[i])/numPulls[i] + float(bias) / (2*numPulls[i])
        if indVal < curtVal:
            rst, indVal = i, curtVal
    return rst

def ucb(numSucs, numPulls, t):
    """
    :numSucs[i]: S_i(t) cumulative rewards in pulling arm i
    :numPulls[i]: N_i(t) total num of pulls of arm i
    :t: total number of rounds of decisions until now
    """
    rst, indVal = 0, float('-inf')
    for i in range(len(numSucs)):
        curtVal = float(numSucs[i])/numPulls[i] + np.sqrt(2*np.log(t)/numPulls[i])
        if indVal < curtVal:
            rst, indVal = i, curtVal
    return rst

def ksd_index(numSucs, numPulls, armSamples,upperbound,cuttoff_value,max_iterations=100,):
    """
    :numSucs: Cummulative Reward
    :numPulls: N(t) total num of pulls of arm
    :samples: samples of arm
    :t: total number of rounds of decisions until now
    """
    precision = 0.001
    value = max(numSucs/numPulls, -1e5)
    kernel_type = 'rbf' 
    h_method = 'dim'
    iterator = 0
    
    armSamples = torch.from_numpy(np.array(armSamples)).float().reshape(-1,1)
    while iterator < max_iterations and upperbound - value > precision:
        iterator += 1
        m = (value + upperbound) * 0.5
        gradients = (armSamples - m)/1

        ksd_value = get_KSD(armSamples, gradients, kernel_type, h_method)
        if ksd_value > cuttoff_value:
            upperbound = m
        else:
            value = m
        
    return (value + upperbound) * 0.5

def ksd_ucb(numSucs,numPulls,samples,t):
    """
    :armMeans[i]: posterior mean of arm i
    :numPulls[i]: N_i(t) total num of pulls of arm i
    :samples[i]: samples of arm i
    :t: total number of rounds of decisions until now
    """
    rst, indVal = 0, float('-inf')
    for i in range(len(numSucs)):
        armSamples = samples[i]
        cuttoff_value = (np.log(t)+10*np.log(np.log(t)))/numPulls[i]
        upperbound = 1
        ### Compute KSD
        curtVal = ksd_index(numSucs[i],numPulls[i],armSamples,upperbound,cuttoff_value,max_iterations=100)
        print("Arm: ",i," KSD: ",curtVal)
        if indVal < curtVal:
            rst, indVal = i, curtVal
    return rst

def ucb_tuned(numSucs, squaredSucs, numPulls, t):
    """
    :numSucs[i]: S_i(t) cumulative rewards in pulling arm i
    :squaredSucs[i]: cumulative squared rewards in pulling arm i
    :numPulls[i]: N_i(t) total num of pulls of arm i
    :t: total number of rounds of decisions until now
    """
    rst, indVal = 0, float('-inf')
    for i in range(len(numSucs)):
        Vi = float(squaredSucs[i]) / numPulls[i] - (float(numSucs[i]) / numPulls[i]) ** 2 + np.sqrt(2*np.log(t)/numPulls[i])
        curtVal = float(numSucs[i])/numPulls[i] + np.sqrt(min(0.25, Vi)*np.log(t)/numPulls[i])
        if indVal < curtVal:
            rst, indVal = i, curtVal
    return rst

def bayes_ucb(c, mus, stds, t, T):
    """
    Reference: line 9-12 from bottom of page 594: http://proceedings.mlr.press/v22/kaufmann12/kaufmann12.pdf
    Prior is Gaussian prior, follows the setup in IDS paper
    :c: hyper parameter, default is 0
    :mus[i]: posterior mean of arm i
    :stds[i]: posterior sigma of arm i 
    :t: total number of rounds of decisions until now
    :T: time horizon of concern
    """
    rst, indVal = 0, float('-inf')
    for i in range(len(mus)):
        curtVal = norm.ppf(q=1-1/(t * np.log(T)**c), loc=mus[i], scale=stds[i])
        if indVal < curtVal:
            rst, indVal = i, curtVal
    return rst

def kl_ucb_gauss(numSucs, numPulls, sigma, t, cKlUcb):
    """ KL-UCB index computation for Gaussian distributions.
    - Note that it does not require any search.
    .. warning:: it works only if the good variance constant is given.
    """
    rst, indVal = 0, float('-inf')
    for i in range(len(numSucs)):
        maxKLValue = (np.log(t) + cKlUcb*np.log(np.log(t))) / numPulls[i]
        curtVal = float(numSucs[i])/numPulls[i] + np.sqrt(2 * sigma**2 * maxKLValue)
        if indVal < curtVal:
            rst, indVal = i, curtVal
    return rst

def gpucb(mus, stds, t, T, delta):
    """
    Implementation of GPUCB, Srinivas (2010) 'Gaussian Process Optimization in the Bandit Setting: No Regret and 
    Experimental Design' for Gaussian Bandit Problems with normal prior
    """
    rst, indVal = 0, float('-inf')
    beta = 2 * np.log(len(mus) * (t * np.pi) ** 2 / (6 * delta))
    for i in range(len(mus)):
        curtVal = mus[i] + stds[i]*np.sqrt(beta)
        if indVal < curtVal:
            rst, indVal = i, curtVal
    return rst

def gpucb_tuned(mus, stds, t, T):
    """
    Implementation of GPUCB, Srinivas (2010) 'Gaussian Process Optimization in the Bandit Setting: No Regret and 
    Experimental Design' for Gaussian Bandit Problems with normal prior
    """
    rst, indVal = 0, float('-inf')
    beta = 0.9 * np.log(t)
    for i in range(len(mus)):
        curtVal = mus[i] + stds[i]*np.sqrt(beta)
        if indVal < curtVal:
            rst, indVal = i, curtVal
    return rst

def kg(mus, stds, sigma, t, T):
    """
    Reference: page 184, eq (14) of "The Knowledge Gradient Algorithm for a General Class of Online Learning Problems"
    """
    temp = [(mus[i], i) for i in range(len(mus))]
    temp.sort()
    cs = [0] * len(mus)
    for i in range(len(mus)):
        ind = temp[i][1]
        if i == len(mus)-1:
            cs[ind] = temp[i-1][0]
        else:
            cs[ind] = temp[-1][0]
    
    rst, indVal = 0, float('-inf')
    for i in range(len(mus)):
        sigmaTuta = np.sqrt(stds[i]**2/(1+sigma**2 / stds[i]**2))
        z = -abs(float(mus[i]-cs[i])/sigmaTuta)
        curtVal = mus[i] + (T-t) * sigmaTuta * (norm.cdf(z) * z + norm.pdf(z))
        if indVal < curtVal:
            rst, indVal = i, curtVal
    return rst

def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(a=indices, size=1, replace=False)[0]

def kgf(x):
    """
    :param x: np.array
    :return: np.array, f(x) as defined in Ryzhov et al. (2010) 'The knowledge gradient algorithm for
    a general class of online learning problems'
    """
    return norm.cdf(x) * x + norm.pdf(x)


def kg_star(mus, stds, sigma, t, T):
    """
    Implementation of Optimized Knowledge Gradient algorithm for Bernoulli Bandit Problems with normal prior
    as described in Kaminski (2015) 'Refined knowledge-gradient policy for learning probabilities'
    Reference: https://github.com/DBaudry/Information_Directed_Sampling/blob/master/GaussianMAB.py
    :mus[i]: posterior mean of arm i
    :stds[i]: posterior sigma of arm i 
    :t: total number of rounds of decisions until now
    :T: time horizon of concern
    """
    nb_arms = len(mus)
    eta = sigma
    sigmas = np.array(stds)
    delta_t = np.array([mus[i] - np.max(list(mus)[:i] + list(mus)[i + 1:]) for i in range(nb_arms)])
    r = (delta_t / sigmas) ** 2
    m_lower = eta / (4 * sigmas ** 2) * (-1 + r + np.sqrt(1 + 6 * r + r ** 2))
    m_higher = eta / (4 * sigmas ** 2) * (1 + r + np.sqrt(1 + 10 * r + r ** 2))
    m_star = np.zeros(nb_arms)
    for arm in range(nb_arms):
        if T - t <= m_lower[arm]:
            m_star[arm] = T - t
        elif (delta_t[arm] == 0) or (m_higher[arm] <= 1):
            m_star[arm] = 1
        else:
            m_star[arm] = np.ceil(0.5 * ((m_lower + m_higher)[arm])).astype(int)  # approximation
    s_m = np.sqrt((m_star + 1) * sigmas ** 2 / ((eta / sigmas) ** 2 + m_star + 1))
    v_m = s_m * kgf(-np.absolute(delta_t / (s_m + 10e-9)))
    arm = rd_argmax(mus - np.max(mus) + (T-t)*v_m)
    return arm
#-----main-----
methods = ["ucb", "ucb_tuned", "gpucb", "gpucb_tuned", "kl_ucb", "bayes_ucb", "kg", "kg_star", "bmle"]
methods = ["ucb","kl_ucb","ksd"]
numMethods = len(methods)
dictResults = {}
sigma = 1 # standar devition of Gaussin distribution
sigmas = [sigma] * len(means)
maxMean = max(means)
numSucs = np.zeros((numMethods, len(means)), dtype=float)
numPulls = np.zeros((numMethods, len(means)), dtype=float)

# params for kl ucb
cKlUcb, eps, precision, maxNumIters = 3, 1e-15, 1e-6, 100
epsilon = 0.25
# params for gpucb
delta = 0.00001
# params for kg_star
numSamples = 100
# params for vids_sample
M, numQSampled, threshold = 10000, 100, 0.99
# only ucb_tuned nees this
squaredSucs = np.zeros(len(means), dtype=float)
# params for bayes_ucb
c=0

allRegrets = np.zeros((numMethods, numExps, T), dtype=float)
allRunningTimes = np.zeros((numMethods, numExps, T), dtype=float)
np.random.seed(seed)
allExperiments = np.random.normal(loc=means,scale=sigmas, size=[numExps, T, len(means)])

run_exp = 1
if run_exp:
    for expInd in tqdm(range(numExps)):
        ksd_samples = [[] for _ in range(len(means))]
        samples_length = 2
        for sample in range(samples_length):
            for i in range(len(means)):
                ksd_samples[i].append(np.random.normal(loc=means[i],scale=sigmas[i]))

        # posterior mean and variance, who needs it: gpucb, gpucb_tuned, kg, kg_star, vids
        # although the rest do not need it, we consider them in initialization, those that don't need will not be udpated
        mu0, s0= 0, 1
        mus = mu0 * np.ones((numMethods, len(means)), dtype=float)
        stds = s0 * np.ones((numMethods, len(means)), dtype=float)
        
        for i in range(len(means)):
            numPulls[:,i] = 1
            reward = allExperiments[expInd,i,i]
            numSucs[:,i] = reward
            allRegrets[:,expInd,i] = maxMean - means[i]
            # below line only for ucb-tuned
            squaredSucs[i] = reward **2
            # below line only for bayes ucb, vids_sample
            x = reward
            mus[:,i] = (sigma ** 2 * mus[:,i] + x * stds[:,i] ** 2) / (sigma ** 2 + stds[:,i] ** 2)
            stds[:,i] = np.sqrt((sigma * stds[:,i]) ** 2 / (sigma ** 2 + stds[:,i] ** 2))

        for t in tqdm(range(len(means)+1, T+1)):
            bias = np.log(t) ** 0.5
            rs = allExperiments[expInd,t-1,:]
            mPos = 0
            
            #----deterministic algorithms----
            # ucb
            startTime = time.time()
            arm = ucb(numSucs[mPos,:], numPulls[mPos,:], t)
            duration = time.time()-startTime
        
            allRunningTimes[mPos][expInd][t-1]=duration
            allRegrets[mPos][expInd][t-1] =  maxMean - means[arm]
            numPulls[mPos][arm] += 1
            numSucs[mPos][arm] += rs[arm]
            mPos += 1
            
            # # ucb_tuned
            # startTime = time.time()
            # arm = ucb_tuned(numSucs[mPos,:], squaredSucs, numPulls[mPos,:], t)
            # duration = time.time()-startTime
            
            # allRunningTimes[mPos][expInd][t-1]=duration
            # allRegrets[mPos][expInd][t-1] = maxMean - means[arm]
            # numPulls[mPos][arm] += 1
            # numSucs[mPos][arm] += rs[arm]
            # squaredSucs[i] += rs[arm] **2
            # mPos += 1
            
            # # gpucb
            # startTime = time.time()
            # arm = gpucb(mus[mPos,:], stds[mPos,:], t, T, delta)
            # duration = time.time()-startTime
        
            # allRunningTimes[mPos][expInd][t-1]=duration
            # allRegrets[mPos][expInd][t-1] =  maxMean - means[arm]
            # numPulls[mPos][arm] += 1
            # numSucs[mPos][arm] += rs[arm]
            # x = rs[arm]
            # mus[mPos, arm] = (sigma ** 2 * mus[mPos, arm] + x * stds[mPos, arm] ** 2) / (sigma ** 2 + stds[mPos, arm] ** 2)
            # stds[mPos, arm] = np.sqrt((sigma * stds[mPos, arm]) ** 2 / (sigma ** 2 + stds[mPos, arm] ** 2))
            # mPos += 1
            
            # """
            # Reference: page 2: http://www.ams.sunysb.edu/~zhu/ams570/Bayesian_Normal.pdf
            # Update posterior mean and std for the chosen arm only
            # """
            
            # #gpucb_tuned
            # startTime = time.time()
            # arm = gpucb_tuned(mus[mPos,:], stds[mPos,:], t, T)
            # duration = time.time()-startTime
        
            # allRunningTimes[mPos][expInd][t-1]=duration
            # allRegrets[mPos][expInd][t-1] =  maxMean - means[arm]
            # numPulls[mPos][arm] += 1
            # numSucs[mPos][arm] += rs[arm]
            # x = rs[arm]
            # mus[mPos, arm] = (sigma ** 2 * mus[mPos, arm] + x * stds[mPos, arm] ** 2) / (sigma ** 2 + stds[mPos, arm] ** 2)
            # stds[mPos, arm] = np.sqrt((sigma * stds[mPos, arm]) ** 2 / (sigma ** 2 + stds[mPos, arm] ** 2))
            # mPos += 1
            
            # kl_ucb
            startTime = time.time()
            arm = kl_ucb_gauss(numSucs[mPos,:], numPulls[mPos,:], sigma, t, cKlUcb)
            duration = time.time()-startTime
        
            allRunningTimes[mPos][expInd][t-1]=duration
            allRegrets[mPos][expInd][t-1] =  maxMean - means[arm]
            numPulls[mPos][arm] += 1
            numSucs[mPos][arm] += rs[arm]
            mPos += 1
            
            # # bayes_ucb
            # startTime = time.time()
            # arm = bayes_ucb(c, mus[mPos,:], stds[mPos,:], t, T)
            # duration = time.time()-startTime
        
            # allRunningTimes[mPos][expInd][t-1]=duration
            # allRegrets[mPos][expInd][t-1] =  maxMean - means[arm]
            # numPulls[mPos][arm] += 1
            # numSucs[mPos][arm] += rs[arm]
            # x = rs[arm]
            # mus[mPos, arm] = (sigma ** 2 * mus[mPos, arm] + x * stds[mPos, arm] ** 2) / (sigma ** 2 + stds[mPos, arm] ** 2)
            # stds[mPos, arm] = np.sqrt((sigma * stds[mPos, arm]) ** 2 / (sigma ** 2 + stds[mPos, arm] ** 2))
            # mPos += 1
            
            # # kg
            # startTime = time.time()
            # arm = kg(mus[mPos,:], stds[mPos,:], sigma, t, T)
            # duration = time.time()-startTime
        
            # allRunningTimes[mPos][expInd][t-1]=duration
            # allRegrets[mPos][expInd][t-1] =  maxMean - means[arm]
            # numPulls[mPos][arm] += 1
            # numSucs[mPos][arm] += rs[arm]
            # x = rs[arm]
            # mus[mPos, arm] = (sigma ** 2 * mus[mPos, arm] + x * stds[mPos, arm] ** 2) / (sigma ** 2 + stds[mPos, arm] ** 2)
            # stds[mPos, arm] = np.sqrt((sigma * stds[mPos, arm]) ** 2 / (sigma ** 2 + stds[mPos, arm] ** 2))
            # mPos += 1
            
            # # kg_star
            # startTime = time.time()
            # arm = kg_star(mus[mPos,:], stds[mPos,:], sigma, t, T)
            # duration = time.time()-startTime
        
            # allRunningTimes[mPos][expInd][t-1]=duration
            # allRegrets[mPos][expInd][t-1] =  maxMean - means[arm]
            # numPulls[mPos][arm] += 1
            # numSucs[mPos][arm] += rs[arm]
            # x = rs[arm]
            # mus[mPos, arm] = (sigma ** 2 * mus[mPos, arm] + x * stds[mPos, arm] ** 2) / (sigma ** 2 + stds[mPos, arm] ** 2)
            # stds[mPos, arm] = np.sqrt((sigma * stds[mPos, arm]) ** 2 / (sigma ** 2 + stds[mPos, arm] ** 2))
            # mPos += 1
            
            # # bmle
            # startTime = time.time()
            # arm = bmle(numSucs[mPos,:], numPulls[mPos,:], t, bias, sigma)
            # duration = time.time()-startTime
        
            # allRunningTimes[mPos][expInd][t-1]=duration
            # allRegrets[mPos][expInd][t-1] =  maxMean - means[arm]
            # numPulls[mPos][arm] += 1
            # numSucs[mPos][arm] += rs[arm]
            # mPos += 1
            
            ### ksd_ucb
            sample_limit = 100
            for arm in range(len(means)):
                if len(ksd_samples[arm]) > sample_limit:
                    ksd_samples[arm] = ksd_samples[arm][-sample_limit:]
                
            startTime = time.time()
            arm = ksd_ucb(numSucs[mPos,:], numPulls[mPos,:], ksd_samples, t)
            ksd_samples[arm].append(rs[arm])
            duration = time.time()-startTime

            allRunningTimes[mPos][expInd][t-1]=duration
            allRegrets[mPos][expInd][t-1] =  maxMean - means[arm]
            numPulls[mPos][arm] += 1
            numSucs[mPos][arm] += rs[arm]
            mPos += 1

    np.save("./allRegrets.npy", allRegrets)
allRegrets = np.load("./allRegrets.npy")
cumRegrets = np.cumsum(allRegrets,axis=2)
import matplotlib.pyplot as plt 
plt.plot(cumRegrets.mean(axis=1)[0,:],label="ucb")
plt.plot(cumRegrets.mean(axis=1)[1,:],label="kl_ucb")
plt.plot(cumRegrets.mean(axis=1)[2,:],label="ksd_ucb")
plt.legend()
plt.savefig("plots/cum_regret.png")

meanRegrets = np.mean(cumRegrets,axis=1)
stdRegrets = np.std(cumRegrets,axis=1)
meanFinalRegret = meanRegrets[:,-1]
stdFinalRegret = stdRegrets[:,-1]
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
finalRegretQuantiles = np.quantile(cumRegrets[:,:,-1], q=quantiles, axis=1)

cumRunningTimes = np.cumsum(allRunningTimes,axis=2)
meanRunningTimes = np.mean(cumRunningTimes,axis=1)
stdRunningTimes = np.std(cumRunningTimes,axis=1)
meanTime = np.sum(allRunningTimes, axis=(1,2))/(T*numExps)
stdTime = np.std(allRunningTimes, axis=(1,2))
runningTimeQuantiles = np.quantile(cumRunningTimes[:,:,-1], q=quantiles, axis=1)



for i in range(len(methods)):
    method = methods[i]
    dictResults[method] = {}
    dictResults[method]["allRegrets"] = np.copy(allRegrets[i])
    dictResults[method]["cumRegrets"] = np.copy(cumRegrets[i])
    dictResults[method]["meanCumRegrets"] = np.copy(meanRegrets[i])
    dictResults[method]["stdCumRegrets"] = np.copy(stdRegrets[i])
    dictResults[method]["meanFinalRegret"] = np.copy(meanFinalRegret[i])
    dictResults[method]["stdFinalRegret"] = np.copy(stdFinalRegret[i])
    dictResults[method]["finalRegretQuantiles"] = np.copy(finalRegretQuantiles[:,i])
    
    
    dictResults[method]["allRunningTimes"] = np.copy(allRunningTimes[i])
    dictResults[method]["cumRunningTimes"] = np.copy(cumRunningTimes[i])
    dictResults[method]["meanCumRunningTimes"] = np.copy(meanRunningTimes[i])
    dictResults[method]["stdCumRunningTimes"] = np.copy(stdRunningTimes[i])
    dictResults[method]["meanTime"] = np.copy(meanTime[i])
    dictResults[method]["stdTime"] = np.copy(stdTime[i])
    dictResults[method]["runningTimeQuantiles"] = np.copy(runningTimeQuantiles[:,i])

    
with open(path + 'ID=' + str(excelID) + '_gaussian_deterministic_algs_seed_' + str(seed) + '.pickle', 'wb') as handle:
    pickle.dump(dictResults, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print out the average cumulative regret all methods
with open(path + 'ID=' + str(excelID) + '_gaussian_deterministic_algs_seed_' + str(seed) + '.pickle', 'rb') as handle:
    dictResults = pickle.load(handle)
for method in dictResults:
    print (method, '--', dictResults[method]["meanFinalRegret"])


# %%
