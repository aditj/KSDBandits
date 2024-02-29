### EM Algorithm using Gaussian Mixture Model from scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm
# Generate synthetic data (for demonstration purposes)
n = 100
true_mixing_coeff = 0.3
uniform_rvs = np.random.rand(n)
X_0 = np.random.normal(0.2,0.1, size=(n))
X_1 = np.random.normal(0.5,0.1, size=(n))
x = np.zeros((n,))
x[uniform_rvs < true_mixing_coeff] = X_0[uniform_rvs < true_mixing_coeff]
x[uniform_rvs >= true_mixing_coeff] = X_1[uniform_rvs >= true_mixing_coeff]
x = x.reshape(-1,1)

# Fit a Gaussian Mixture Model
def fit_gmm(x, n_components=2, max_iter=100, n_init=1):
    mean_prior = np.array([0.5])
    mean_precision_prior = 10
    weight_concentration_prior = 1
    bgm = BayesianGaussianMixture(n_components=2, max_iter=100, n_init=1,weight_concentration_prior=weight_concentration_prior, weight_concentration_prior_type='dirichlet_distribution',mean_prior=mean_prior,covariance_type='diag')
    bgm.fit(x.reshape(-1,1))
    weights = bgm.weights_
    means = bgm.means_
    mean_variance = bgm.mean_precision_**-1
    return weights, means, mean_variance

def sample_gmm_parameters(weights, means, mean_variance,std=1, n_samples=1000):
    n_components = len(weights)
    samples = np.zeros((n_samples,3,n_components))
    samples[:,0] = np.random.dirichlet(weights)
    samples[:,1,:] = np.random.normal(means.flatten(), np.sqrt(mean_variance), size=(n_samples,n_components))
    samples[:,2] = std
    
    return samples

def gradient_gmm_parameters(weights, means, std, x):
    n_components = len(weights)
    n_samples = len(x)
    gradient = np.zeros((n_samples,n_components))
    ## print types of weights, means, and mean_variance and norm.pdf
    for i in range(n_components):
        for j in range(n_samples):
            gradient[j,i] = weights[i]*norm.pdf(x[j], means[i], np.sqrt(std[i]))*np.array(x[j]-means[i])/(np.sum(weights*norm.pdf(x[j], means, np.sqrt(std)))*std[i]**2)
    return gradient

# weights, means, mean_variance = fit_gmm(x)
# samples = sample_gmm_parameters(weights, means, mean_variance)
# print(samples)

# print(gradient_gmm_parameters(weights, samples[0,1],samples[0,2], x))
