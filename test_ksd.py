import numpy as np
import torch
from ksd import get_KSD

kernel_type = 'rbf'
h_method = 'dim'

true_mean = 1
sample_lengths = [10, 100, 1000]
n_mc = 100
for sample_length in sample_lengths:
    ksd_estimates = []

    for i in range(n_mc):
        samples = np.random.normal(true_mean, 2,  sample_length).reshape(-1,1)
        samples = torch.from_numpy(samples).float()
        mean = true_mean
        ### Compute gradient of log likelihood of Gaussian with parameters mean and sigma at samples
        gradients = (samples - mean)/1
        ### Compute KSD
        ksd_estimate = get_KSD(samples, gradients, kernel_type, h_method)
        ksd_estimates.append(ksd_estimate)
    print(np.mean(ksd_estimates),np.var(ksd_estimates))

