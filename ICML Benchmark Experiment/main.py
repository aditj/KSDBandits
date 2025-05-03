import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Callable
from sklearn.mixture import GaussianMixture # Added for EM
import matplotlib.pyplot as plt
import tqdm 
class MultimodalArm:
    """Represents a single arm with a Gaussian Mixture Model (GMM) distribution."""
    def __init__(self, means: List[float], variances: List[float], weights: List[float]):
        if not (len(means) == len(variances) == len(weights)):
            raise ValueError("Means, variances, and weights must have the same length.")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.")

        self.means = np.array(means)
        self.variances = np.array(variances)
        self.weights = np.array(weights)
        self.num_components = len(means)

    def sample(self) -> float:
        """Sample a reward from the GMM distribution."""
        component = np.random.choice(self.num_components, p=self.weights)
        return np.random.normal(loc=self.means[component], scale=np.sqrt(self.variances[component]))

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return the parameters of the distribution."""
        return {"means": self.means, "variances": self.variances, "weights": self.weights}

class BanditInstance:
    """
    Represents the multi-armed bandit problem instance.
    Initializes arms with multimodal distributions (GMMs).
    """
    def __init__(self, arm_params: List[Dict[str, List[float]]]):
        """
        Initializes the bandit instance.

        Args:
            arm_params: A list where each element is a dictionary
                        {'means': [...], 'variances': [...], 'weights': [...]}
                        defining the parameters for one arm's GMM.
        """
        self.arms = [MultimodalArm(**params) for params in arm_params]
        self.num_arms = len(self.arms)

    def pull(self, arm_index: int) -> float:
        """
        Pulls the specified arm and returns a reward.

        Args:
            arm_index: The index of the arm to pull.

        Returns:
            The reward sampled from the arm's distribution.

        Raises:
            IndexError: If arm_index is out of bounds.
        """
        if not 0 <= arm_index < self.num_arms:
            raise IndexError("Arm index out of bounds.")
        return self.arms[arm_index].sample()

    def get_true_params(self) -> List[Dict[str, np.ndarray]]:
         """Returns the true parameters for all arms."""
         return [arm.get_params() for arm in self.arms]

class Metric(ABC):
    """
    Abstract base class for metrics used to evaluate/rank arms
    based on their distribution parameters.
    """
    @abstractmethod
    def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
        """
        Computes a score for an arm given its estimated or true parameters.
        Higher scores are generally better.

        Args:
            means: Array of means for the GMM components.
            variances: Array of variances for the GMM components.
            weights: Array of weights for the GMM components.

        Returns:
            A scalar score representing the arm's quality according to this metric.
        """
        pass

    def get_optimal_arm_index(self, bandit_instance: BanditInstance) -> int:
        """
        Determines the optimal arm based on the true parameters and the metric.

        Args:
            bandit_instance: The BanditInstance containing the true arm parameters.

        Returns:
            The index of the best arm according to this metric.
        """
        true_params = bandit_instance.get_true_params()
        scores = [self.compute_score(**params) for params in true_params]
        return np.argmax(scores)

    def calculate_cumulative_regret(
        self,
        bandit_instance: BanditInstance,
        pulled_arms: List[int],
        rewards: List[float]  # Keep rewards for potential future use, though not needed for expected regret
    ) -> np.ndarray:
        """
        Calculates the cumulative regret over time based on the sequence of pulls.

        Args:
            bandit_instance: The BanditInstance containing the true arm parameters.
            pulled_arms: List of indices of the arms pulled at each step.
            rewards: List of rewards received at each step.

        Returns:
            A numpy array where element i is the cumulative regret up to step i.
        """
        if not pulled_arms:
            return np.array([])

        optimal_arm_index = self.get_optimal_arm_index(bandit_instance)
        optimal_arm_params = bandit_instance.get_true_params()[optimal_arm_index]
        optimal_expected_reward = np.sum(optimal_arm_params['weights'] * optimal_arm_params['means'])

        true_params = bandit_instance.get_true_params()
        expected_rewards_pulled = np.array([
            np.sum(true_params[arm_idx]['weights'] * true_params[arm_idx]['means'])
            for arm_idx in pulled_arms
        ])

        # Instantaneous regret at each step: E[optimal] - E[pulled_i]
        instantaneous_regret = optimal_expected_reward - expected_rewards_pulled

        # Cumulative regret over time
        cumulative_regret_over_time = np.cumsum(instantaneous_regret)

        return cumulative_regret_over_time


class BanditAlgorithm(ABC):
    """
    Abstract base class for bandit algorithms.
    """
    def __init__(self, num_arms: int, metric: Metric):
        """
        Initializes the bandit algorithm.

        Args:
            num_arms: The number of arms in the bandit problem.
            metric: The metric object used for evaluating arms.
        """
        if num_arms <= 0:
            raise ValueError("Number of arms must be positive.")
        self.num_arms = num_arms
        self.metric = metric
        self.t = 0  # Timestep counter
        self.counts = np.zeros(num_arms, dtype=int) # Number of times each arm was pulled
        self.history = {i: [] for i in range(num_arms)} # Stores rewards for each arm

    def update(self, arm_index: int, reward: float):
        """
        Updates the algorithm's internal state after pulling an arm.

        Args:
            arm_index: The index of the arm that was pulled.
            reward: The reward received from pulling the arm.
        """
        if not 0 <= arm_index < self.num_arms:
            raise IndexError("Arm index out of bounds.")
        self.counts[arm_index] += 1
        self.history[arm_index].append(reward)
        self.t += 1

    @abstractmethod
    def select_arm(self) -> int:
        """
        Selects an arm to pull based on the algorithm's strategy and history.

        Returns:
            The index of the arm to pull next.
        """
        pass

class ExploreThenCommitEM(BanditAlgorithm):
    """
    Bandit algorithm using Explore-Then-Commit strategy with EM for parameter estimation.
    1. Explores each arm T_e times.
    2. Estimates GMM parameters using EM.
    3. Commits to the arm with the best score based on the metric and estimated parameters.
    """
    def __init__(self, num_arms: int, metric: Metric, explore_rounds: int, n_components_to_fit: int = 2):
        """
        Initializes the algorithm.

        Args:
            num_arms: Number of arms.
            metric: Metric object for scoring arms.
            explore_rounds: Number of times (T_e) to pull each arm during exploration.
            n_components_to_fit: The number of Gaussian components to fit using EM.
                                 This might differ from the true number of components.
        """
        super().__init__(num_arms, metric)
        if explore_rounds <= 0:
            raise ValueError("Explore rounds (T_e) must be positive.")
        self.explore_rounds = explore_rounds
        self.n_components_to_fit = n_components_to_fit
        self.exploration_phase_steps = self.explore_rounds * self.num_arms
        self.estimated_params = None # To store estimated parameters after exploration
        self.committed_arm = None    # To store the chosen arm after exploration

    def _estimate_parameters(self):
        """Estimate GMM parameters for each arm using EM based on history."""
        self.estimated_params = []
        scores = []

        for i in range(self.num_arms):
            rewards = self.history[i]
            if len(rewards) < self.n_components_to_fit:
                # Handle cases with insufficient data - assign default low score or simple estimate
                estimated = {'means': np.array([0]), 'variances': np.array([1]), 'weights': np.array([1.0])}
                score = -np.inf # Assign a very low score
            else:
                try:
                    rewards_array = np.array(rewards).reshape(-1, 1)
                    gmm = GaussianMixture(n_components=self.n_components_to_fit,
                                            covariance_type='diag', # Use diagonal covariance matrices
                                            random_state=0, # For reproducibility
                                            n_init=3 # Number of initializations to run
                                           )
                    gmm.fit(rewards_array)

                    # Ensure parameters are sorted by mean for consistency if needed by metric, although not strictly required by GMM
                    # sort_indices = np.argsort(gmm.means_.flatten())
                    # means = gmm.means_.flatten()[sort_indices]
                    # variances = gmm.covariances_.flatten()[sort_indices]
                    # weights = gmm.weights_[sort_indices]

                    estimated = {
                        'means': gmm.means_.flatten(),
                        'variances': gmm.covariances_.flatten(),
                        'weights': gmm.weights_
                    }
                    score = self.metric.compute_score(**estimated)

                except Exception as e:
                    print(f"  Arm {i}: Error during GMM fitting: {e}. Assigning low score.")
                    estimated = {'means': np.array([np.mean(rewards) if rewards else 0]),
                                 'variances': np.array([np.var(rewards) if rewards else 1]),
                                 'weights': np.array([1.0])}
                    score = -np.inf # Assign a very low score if EM fails

            self.estimated_params.append(estimated)
            scores.append(score)

        self.committed_arm = np.argmax(scores)

    def select_arm(self) -> int:
        """
        Selects an arm based on the current phase (exploration or commitment).
        """
        # Exploration Phase
        if self.t < self.exploration_phase_steps:
            # Pull arms sequentially
            return self.t % self.num_arms

        # Estimation and Commitment Phase
        else:
            # Estimate parameters exactly once after exploration finishes
            if self.committed_arm is None:
                self._estimate_parameters()

            # Commit to the best arm based on estimation
            return self.committed_arm

# --- Extended Metric Implementations ---
class MeanBasedMetric(Metric):
        """A simple metric based on the expected value of the GMM."""
        def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
            # E[X] = sum(weight_i * mean_i)
            return np.sum(weights * means)

class MaxMeanMetric(Metric):
    """Metric that returns the maximum of the component means."""
    def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
        return float(np.max(means))

class WeightedMeanMetric(Metric):
    """Metric that returns a weighted sum of component means with user-defined weights."""
    def __init__(self, component_weights: List[float]):
        self.component_weights = np.array(component_weights)
        if not np.isclose(np.sum(self.component_weights), 1.0):
            raise ValueError("component_weights must sum to 1.")

    def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
        if len(means) != len(self.component_weights):
            raise ValueError("Mismatch between component_weights and number of mixture components.")
        return float(np.dot(self.component_weights, means))

class SharpeRatioMetric(Metric):
    """Metric that computes the Sharpe ratio (mean/std) of the GMM."""
    def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
        mu = float(np.sum(weights * means))
        var = float(np.sum(weights * (variances + means**2)) - mu**2)
        std = np.sqrt(var) if var > 0 else np.inf
        return float(mu / std) if std > 0 else 0.0

class SpreadMetric(Metric):
    """Metric that computes (max mean - min mean) among components."""
    def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
        return float(np.max(means) - np.min(means))

class CVaRMetric(Metric):
    """Metric that approximates negative CVaR at level alpha (to minimize tail risk)."""
    def __init__(self, alpha: float = 0.05, n_samples: int = 10000):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1).")
        self.alpha = alpha
        self.n_samples = n_samples

    def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
        # Monte Carlo estimate of CVaR
        samples = np.empty(self.n_samples)
        for i in range(self.n_samples):
            comp = np.random.choice(len(means), p=weights)
            samples[i] = np.random.normal(means[comp], np.sqrt(variances[comp]))
        threshold = np.quantile(samples, self.alpha)
        cvar = samples[samples <= threshold].mean() if np.any(samples <= threshold) else threshold
        # we want to minimize CVaR, so return negative
        return float(-cvar)

class ModalEntropyMetric(Metric):
    """Metric that computes entropy of the mixture weights (diversity of modes)."""
    def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
        ent = -np.sum(weights * np.log(weights + 1e-12))
        return float(ent)

class GiniIndexMetric(Metric):
    """Metric that computes the Gini index over component means weighted by mixture weights."""
    def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
        mu = float(np.sum(weights * means))
        # Compute pairwise weighted absolute differences
        diff = 0.0
        for i in range(len(means)):
            for j in range(len(means)):
                diff += weights[i] * weights[j] * abs(means[i] - means[j])
        gini = diff / (2 * mu) if mu != 0 else 0.0
        return float(gini)

class ModeSeparationMetric(Metric):
    """Metric that returns the minimum separation between component means."""
    def compute_score(self, means: np.ndarray, variances: np.ndarray, weights: np.ndarray) -> float:
        if len(means) < 2:
            return 0.0
        sorted_means = np.sort(means)
        separations = np.abs(np.diff(sorted_means))
        return float(np.min(separations))

class RandomAlgorithm(BanditAlgorithm):
    def select_arm(self) -> int:
        return np.random.randint(self.num_arms)

# Add CVaR-UCB algorithm
class CVaRUCBAlgorithm(BanditAlgorithm):
    """CVaR-UCB algorithm (Galichet et al., 2013)."""
    def __init__(self, num_arms: int, metric: Metric, alpha: float = 0.05):
        super().__init__(num_arms, metric)
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1).")
        self.alpha = alpha

    def _empirical_cvar(self, samples: List[float]) -> float:
        arr = np.sort(np.array(samples))
        n = len(arr)
        k = int(np.ceil(self.alpha * n))
        if k < 1:
            k = 1
        u = arr[k - 1]
        reg = np.sum(np.maximum(0, u - arr))
        return float(u + (1 / (self.alpha * n)) * reg)

    def select_arm(self) -> int:
        # Ensure each arm is pulled at least once
        for i in range(self.num_arms):
            if self.counts[i] == 0:
                return i
        T = max(self.t, 1)
        values = []
        for i in range(self.num_arms):
            samples = self.history[i]
            ni = len(samples)
            cvar_hat = self._empirical_cvar(samples)
            bound = np.sqrt((2 * np.log(T)) / (self.alpha ** 2 * ni))
            values.append(cvar_hat - bound)
        return int(np.argmax(values))

# Add Sharpe-Ratio-UCB algorithm
class SharpeUCBAlgorithm(BanditAlgorithm):
    """Sharpe-Ratio-UCB algorithm (Khurshid et al., 2024)."""
    def __init__(self, num_arms: int, metric: Metric, r: float = 0.0, eps: float = 1e-6, c: float = 2.0):
        super().__init__(num_arms, metric)
        if c <= 0:
            raise ValueError("c must be positive.")
        self.r = r
        self.eps = eps
        self.c = c

    def select_arm(self) -> int:
        # Ensure each arm is pulled at least once
        for i in range(self.num_arms):
            if self.counts[i] == 0:
                return i
        T = max(self.t, 1)
        values = []
        for i in range(self.num_arms):
            samples = np.array(self.history[i])
            ni = len(samples)
            mu = samples.mean()
            var = samples.var(ddof=0)
            sharpe = (mu - self.r) / np.sqrt(var + self.eps)
            bonus = np.sqrt(self.c * np.log(T) / ni)
            values.append(sharpe + bonus)
        return int(np.argmax(values))

# Add Rank-Dependent Expected-Utility (RDEU) algorithm
class RDEUAlgorithm(BanditAlgorithm):
    """Rank-Dependent Expected-Utility bandit algorithm."""
    def __init__(self,
                 num_arms: int,
                 metric: Metric,
                 u: Callable[[float], float],
                 w: Callable[[float], float],
                 beta: float = 2.0):
        super().__init__(num_arms, metric)
        if beta <= 0:
            raise ValueError("beta must be positive.")
        self.u = u
        self.w = w
        self.beta = beta

    def select_arm(self) -> int:
        # Ensure each arm is pulled at least once
        for i in range(self.num_arms):
            if self.counts[i] == 0:
                return i
        T = max(self.t, 1)
        values = []
        for i in range(self.num_arms):
            samples = np.sort(np.array(self.history[i]))
            n = len(samples)
            # Compute RDEU: sum over order statistics
            utility = 0.0
            for k_idx, x in enumerate(samples):
                k = k_idx + 1
                p1 = (n - k + 1) / n
                p0 = (n - k) / n
                delta = self.w(p1) - self.w(p0)
                utility += self.u(x) * delta
            bonus = np.sqrt(self.beta * np.log(T) / n)
            values.append(utility + bonus)
        return int(np.argmax(values))

# Add Mean-based UCB algorithm
class MeanUCBAlgorithm(BanditAlgorithm):
    """Mean-based UCB (upper confidence bound) algorithm."""
    def __init__(self, num_arms: int, metric: Metric):
        super().__init__(num_arms, metric)

    def select_arm(self) -> int:
        # Ensure each arm is pulled at least once
        for i in range(self.num_arms):
            if self.counts[i] == 0:
                return i
        T = max(self.t, 1)
        values = []
        for i in range(self.num_arms):
            samples = np.array(self.history[i])
            ni = len(samples)
            mu = float(samples.mean())
            bonus = np.sqrt(2 * np.log(T) / ni)
            values.append(mu + bonus)
        return int(np.argmax(values))

if __name__ == "__main__":
    num_trials = 100
    n_steps = 1000
    num_arms = 3
    num_components = 4  # number of modes per arm

    # Define metrics
    metrics = [
        MeanBasedMetric(),
        MaxMeanMetric(),
        WeightedMeanMetric([0.1, 0.2, 0.3, 0.4]),
        SharpeRatioMetric(),
        SpreadMetric(),
        CVaRMetric(alpha=0.05, n_samples=5000),
        ModalEntropyMetric(),
        GiniIndexMetric(),
        ModeSeparationMetric()
    ]

    # Define algorithms (name, class)
    algorithms = [
        ('Random', RandomAlgorithm),
        ('ExploreThenCommitEM', ExploreThenCommitEM),
        ('CVaRUCB', CVaRUCBAlgorithm),
        ('SharpeUCB', SharpeUCBAlgorithm),
        ('RDEU', RDEUAlgorithm),
        ('MeanUCB', MeanUCBAlgorithm)
    ]

    # Storage: {metric_name: {algo_name: [cumulative_regret_array, ...]}}
    regrets = {m.__class__.__name__: {a[0]: [] for a in algorithms} for m in metrics}

    for trial in tqdm.tqdm(range(num_trials)):
        # Randomly initialize a bandit instance with multimodal arms
        arm_params = []
        for _ in range(num_arms):
            means = np.random.normal(loc=0, scale=5, size=num_components)
            variances = np.random.uniform(low=0.5, high=2.0, size=num_components)
            weights = np.random.dirichlet(alpha=np.ones(num_components))
            arm_params.append({
                'means': means.tolist(),
                'variances': variances.tolist(),
                'weights': weights.tolist()
            })
        bandit = BanditInstance(arm_params)

        for metric in metrics:
            for alg_name, AlgClass in algorithms:
                # Instantiate algorithm
                if alg_name == 'ExploreThenCommitEM':
                    algo = AlgClass(num_arms, metric, explore_rounds=50, n_components_to_fit=num_components)
                elif alg_name == 'CVaRUCB':
                    algo = AlgClass(num_arms, metric, alpha=0.05)
                elif alg_name == 'SharpeUCB':
                    algo = AlgClass(num_arms, metric, r=0.0, eps=1e-6, c=2.0)
                elif alg_name == 'RDEU':
                    algo = AlgClass(num_arms, metric, u=lambda x: x, w=lambda p: p, beta=2.0)
                elif alg_name == 'MeanUCB':
                    algo = AlgClass(num_arms, metric)
                else:
                    algo = AlgClass(num_arms, metric)

                pulled_hist = []
                rewards_hist = []
                for t in range(n_steps):
                    arm = algo.select_arm()
                    rew = bandit.pull(arm)
                    algo.update(arm, rew)
                    pulled_hist.append(arm)
                    rewards_hist.append(rew)

                cum_reg = metric.calculate_cumulative_regret(bandit, pulled_hist, rewards_hist)
                regrets[metric.__class__.__name__][alg_name].append(cum_reg)

    print("Completed experiments across metrics and algorithms.")

    # Compute average cumulative regret across trials
    avg_regrets = {
        metric: {
            alg: np.mean(np.vstack(regs), axis=0)
            for alg, regs in alg_dict.items()
        }
        for metric, alg_dict in regrets.items()
    }

    # Optional: plot average regret curves if matplotlib is installed
    try:
        import matplotlib.pyplot as plt
        # Create a 3x3 grid of subplots for the 9 metrics
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
        axes = axes.flatten()
        for idx, (metric_name, alg_dict) in enumerate(avg_regrets.items()):
            ax = axes[idx]
            for alg_name, avg in alg_dict.items():
                ax.plot(avg, label=alg_name)
            ax.set_title(metric_name)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Cumulative Regret')
            ax.legend(fontsize='small')
            ax.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Install matplotlib to plot average regrets.")
    