import numpy as np

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
    sample = np.random.normal(true_means[arm_idx,component_idx],np.sqrt(true_variance[component_idx]))
    return sample
