
# coding: utf-8

# In[ ]:

import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import time
import pickle
import torch
from em_algorithm import fit_gmm, sample_gmm_parameters, gradient_gmm_parameters
from ksd import get_KSD

"""
Deterministic algorithms include
"ucb", "ucb_tuned", "gpucb", "gpucb_tuned", "kl_ucb", "bayes_ucb", "kg", "kg_star", "bmle"
"""



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

def ksd_index(numSucs, numPulls, armSamples,upperbound,cuttoff_value,samples_em):
    """
    :numSucs: Cummulative Reward
    :numPulls: N(t) total num of pulls of arm
    :samples: samples of arm
    :t: total number of rounds of decisions until now
    """

    kernel_type = 'rbf' 
    h_method = 'dim'
    max_value = 0
    armSamples = torch.from_numpy(np.array(armSamples,dtype=float)).reshape(-1,1)
    weights,means_parameters,mean_variance_parameters = fit_gmm(np.array(samples_em))
    n_sample_parameters = 15
    samples_parameters = sample_gmm_parameters(weights,means_parameters,mean_variance_parameters,n_samples=n_sample_parameters)
    for i in range(n_sample_parameters):
        gradients = gradient_gmm_parameters(samples_parameters[i,0],samples_parameters[i,1],samples_parameters[i,2],armSamples)
        gradients = torch.from_numpy(gradients).float().reshape(-1,1)
        ksd_value = get_KSD(armSamples, gradients, kernel_type, h_method)

        if ksd_value <= cuttoff_value and ksd_value>max_value:
            max_value = ksd_value

        
    return max_value

def ksd_ucb(numSucs,numPulls,samples,t,samples_em, sigma=1):
    """
    :armMeans[i]: posterior mean of arm i
    :numPulls[i]: N_i(t) total num of pulls of arm i
    :samples[i]: samples of arm i
    :t: total number of rounds of decisions until now
    """
    ### Algorithm: 
    #### 1. Have a prior over the hyperparameters of the gaussian mixture
    #### 2. For each arm sample hyperparameters and compute the KSD index using this as the testing distribution
            ####  2.1 
    #### 3. Choose the arm with the highest KSD index 
    #### 4. Update the posterior over the hyperparameters of the gaussian mixture

    rst, indVal = 0, float('-inf')
    for i in range(len(numSucs)):
        armSamples = samples[i]
        cuttoff_value = (np.log(t)+5*np.log(np.log(t)))/numPulls[i]
        upperbound = 1
        ### Compute KSD
        curtVal = float(numSucs[i])/numPulls[i] +  np.sqrt(2*sigma**2*ksd_index(numSucs[i],numPulls[i],armSamples,upperbound,cuttoff_value,samples_em = samples_em[i]))
        print("Arm: ",i," KSD: ",curtVal - float(numSucs[i])/numPulls[i])
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

def reward_model(parameter_set_1,parameter_set_2,mixing_coeff,len_sample):
    """
    :parameter_set_1: parameters of the first gaussian distribution
    :parameter_set_2: parameters of the second gaussian distribution
    :mixing_coeff: mixing coefficient
    """
    num_arms = len(parameter_set_1)
    samples = np.zeros((num_arms,len_sample))
    for arm in range(num_arms):
        uniform_sample = np.random.rand(len_sample)
        sample_1 = np.random.normal(parameter_set_1[arm],1,len_sample)
        sample_2 = np.random.normal(parameter_set_2[arm],1,len_sample)
        samples[arm,uniform_sample < mixing_coeff] = sample_1[uniform_sample < mixing_coeff]
        samples[arm,uniform_sample >= mixing_coeff] = sample_2[uniform_sample >= mixing_coeff]
    
    return samples.T
    
#-----inputs-----
path = "plots" # directory to save results
means_1 = [0.41, 0.72]#, 0.66, 0.43]#, 0.58] # 0.65, 0.48, 0.67, 0.59, 0.63] # means of Gaussian distribution
means_2 = [0.9,0.1]
excelID = 16 # excelID is used to dfferentiate which set of probs to be tested
seed = 46 # seed number for reproducibility and ensure the data prepared in advance is the same for all parallel programs
numExps = 10 # total number of trials
T = int(1e3) # time horizon T

methods = []
methods += ["ucb","kl_ucb"]
methods += ["ksd_ucb"]
numMethods = len(methods)
dictResults = {}
sigma = 1 # standar devition of Gaussin distribution
n_arms = len(means_1)
sigmas = [sigma] * len(means_1)
maxMean = max(np.array(means_1)*0.25 + np.array(means_2)*0.75)
numSucs = np.zeros((numMethods,n_arms), dtype=float)
numPulls = np.zeros((numMethods, n_arms), dtype=float)

# params for kl ucb
cKlUcb, eps, precision, maxNumIters = 0, 1e-15, 1e-6, 100
epsilon = 0.25
# params for gpucb
delta = 0.00001
# params for kg_star
numSamples = 1
# params for vids_sample
M, numQSampled, threshold = 10000, 100, 0.99
# only ucb_tuned nees this
squaredSucs = np.zeros(n_arms, dtype=float)
# params for bayes_ucb
c=0

allRegrets = np.zeros((numMethods, numExps, T), dtype=float)
allRunningTimes = np.zeros((numMethods, numExps, T), dtype=float)
np.random.seed(seed)
allExperiments = np.zeros((numExps, T, n_arms), dtype=float)
for expInd in range(numExps):
    allExperiments[expInd] = reward_model(means_1,means_2,0.25,T)
run_exp = 1
if run_exp:
    for expInd in tqdm(range(numExps)):
        em_samples = [[] for _ in range(len(means_1))]
        ksd_samples = [[] for _ in range(len(means_1))]
        samples_length = 2
        for sample in range(samples_length):
            for i in range(len(means_1)):
                ksd_samples[i].append(reward_model(means_1,means_2,0.25,1)[0,i])
                em_samples[i].append(reward_model(means_1,means_2,0.25,1)[0,i])
        # posterior mean and variance, who needs it: gpucb, gpucb_tuned, kg, kg_star, vids
        # although the rest do not need it, we consider them in initialization, those that don't need will not be udpated
        mu0, s0= 0, 1
        mus = mu0 * np.ones((numMethods, len(means_1)), dtype=float)
        stds = s0 * np.ones((numMethods, len(means_1)), dtype=float)
        
        for i in range(len(means_1)):
            numPulls[:,i] = 1
            reward = allExperiments[expInd,i,i]
            numSucs[:,i] = reward
            allRegrets[:,expInd,i] = maxMean - means_1[i]*0.25 - means_2[i]*0.75
            # below line only for ucb-tuned
            squaredSucs[i] = reward **2
            # below line only for bayes ucb, vids_sample
            x = reward
            mus[:,i] = (sigma ** 2 * mus[:,i] + x * stds[:,i] ** 2) / (sigma ** 2 + stds[:,i] ** 2)
            stds[:,i] = np.sqrt((sigma * stds[:,i]) ** 2 / (sigma ** 2 + stds[:,i] ** 2))

        for t in tqdm(range(len(means_1)+1, T+1)):
            bias = np.log(t) ** 0.5
            rs = allExperiments[expInd,t-1,:]
            mPos = 0
            
            #----deterministic algorithms----
            # ucb
            startTime = time.time()
            arm = ucb(numSucs[mPos,:], numPulls[mPos,:], t)
            duration = time.time()-startTime
        
            allRunningTimes[mPos][expInd][t-1]=duration
            allRegrets[mPos][expInd][t-1] =  maxMean - means_1[arm]*0.25 - means_2[arm]*0.75
            print("UCB: ",allRegrets[mPos][expInd][t-1])
            numPulls[mPos][arm] += 1
            numSucs[mPos][arm] += rs[arm]
            mPos += 1
            
            # kl_ucb
            startTime = time.time()
            arm = kl_ucb_gauss(numSucs[mPos,:], numPulls[mPos,:], sigma, t, cKlUcb)
            duration = time.time()-startTime
        
            allRunningTimes[mPos][expInd][t-1]=duration
            allRegrets[mPos][expInd][t-1] =  maxMean - means_1[arm]*0.25 - means_2[arm]*0.75
            numPulls[mPos][arm] += 1
            numSucs[mPos][arm] += rs[arm]
            mPos += 1
            
            
            ### ksd_ucb
            sample_limit = 40
            for arm in range(len(means_1)):
                if len(ksd_samples[arm]) > sample_limit:
                    ksd_samples[arm] = ksd_samples[arm][-sample_limit:]
                
            startTime = time.time()
            arm = ksd_ucb(numSucs[mPos,:], numPulls[mPos,:], ksd_samples, t,em_samples)
            ksd_samples[arm].append(rs[arm])
            em_samples[arm].append(rs[arm])
            duration = time.time()-startTime

            allRunningTimes[mPos][expInd][t-1]=duration
            allRegrets[mPos][expInd][t-1] =  maxMean  - means_1[arm]*0.25 - means_2[arm]*0.75
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
