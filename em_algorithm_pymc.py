import numpy as np
import pymc3 as pm
import theano.tensor as tt

# Generate synthetic data (for demonstration purposes)
n = 1000
true_mixing_coeff = 0.4
uniform_rvs = np.random.rand(n)
X_0 = np.random.normal(0.2,0.1, size=(n))
X_1 = np.random.normal(0.5,0.1, size=(n))
x = np.zeros(n)
x[uniform_rvs < true_mixing_coeff] = X_0[uniform_rvs < true_mixing_coeff]
x[uniform_rvs >= true_mixing_coeff] = X_1[uniform_rvs >= true_mixing_coeff]

import matplotlib.pyplot as plt
plt.hist(x, bins=30, density=True)
plt.savefig('hist.png')

with pm.Model() as model:
    # Prior for the mixing coefficient
    pi = pm.Dirichlet('pi', a=np.array([1., 1.]), shape=2)
    
    # Priors for the means of the two Gaussian distributions
    mu = pm.Normal('mu', mu=np.array([0.5, 0.5]), sd=np.array([0.5, 0.5]), shape=2)
    
    # Priors for the standard deviations of the two Gaussian distributions
    sigma = pm.Deterministic('sigma',tt.pow(np.array([0.1,0.1]),1))
    
    # Likelihood (sampling distribution) of observations
    obs = pm.NormalMixture('obs', w=pi, mu=mu, sd=sigma, observed=x)
    approx = pm.fit(1000, method='advi')
    # Inference
    trace = approx.sample(1000)

# You can then inspect the trace to get your parameter estimates
print(pm.summary(trace))