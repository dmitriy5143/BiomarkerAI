import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.cross_decomposition import PLSRegression
from pipeline.fitness_evaluator import FitnessEvaluator
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from joblib import parallel_backend
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.core.termination import Termination

class AdaptiveMySampling:
    def __init__(self, X, y, sss, accuracy_scorer, find_optimal_n_components, adaptation_rate=0.1, update_frequency=3, min_features=5, n_generations=25, initial_top_k=0.5, final_top_k=0.1):
        self.X_full = X
        self.y_full = y
        self.sss = sss
        self.accuracy_scorer = accuracy_scorer
        self.find_optimal_n_components = find_optimal_n_components
        self.n_features = X.shape[1]
        self.adaptation_rate = adaptation_rate
        self.update_frequency = update_frequency
        self.generation = 0
        self.min_features = min_features
        self.n_generations = n_generations
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
        self.update_feature_importance()
        self.feature_probs = np.clip(self.feature_importance, 0.01, 0.99)

    def calculate_vip_scores(self, X, y):
        n_components = self.find_optimal_n_components(X, y, self.sss, self.accuracy_scorer)

        model = PLSRegression(n_components=n_components)
        model.fit(X, y)
        t = model.x_scores_
        w = model.x_weights_
        q = model.y_loadings_
        features_, _ = w.shape
        vip = np.zeros(shape=(features_,))
        inner_sum = np.diag(t.T @ t @ q.T @ q)
        SS_total = np.sum(inner_sum)
        vip = np.sqrt(features_*(w**2 @ inner_sum)/ SS_total)
        return vip

    def update_feature_importance(self):
        vip_scores = self.calculate_vip_scores(self.X_full, self.y_full)
        self.feature_importance = vip_scores / np.sum(vip_scores)

    def adapt_probabilities(self, population):
        progress = self.generation / self.n_generations
        top_k_fraction = self.initial_top_k - (self.initial_top_k - self.final_top_k) * progress
        top_k_fraction = max(self.final_top_k, min(self.initial_top_k, top_k_fraction))
        top_k = max(1, int(len(population) * top_k_fraction))

        fitness_scores = np.array([ind.F[0] for ind in population])
        top_indices = np.argsort(fitness_scores)[-top_k:]
        selected_individuals = [population[i].X for i in top_indices]

        feature_counts = np.sum(selected_individuals, axis=0)
        feature_freq = feature_counts / top_k

        elite_influence = 0.1
        self.feature_probs = (1 - self.adaptation_rate) * self.feature_probs + \
                            self.adaptation_rate * (elite_influence * feature_freq + (1 - elite_influence) * self.feature_importance)
        self.feature_probs = np.clip(self.feature_probs, 0.01, 0.99)

    def compute_entropy(self, population):
        feature_presence = np.array([ind.X for ind in population])
        feature_freq = np.mean(feature_presence, axis=0)

        epsilon = 1e-10
        entropy_per_feature = -feature_freq * np.log2(feature_freq + epsilon) - \
                              (1 - feature_freq) * np.log2(1 - feature_freq + epsilon)
        mean_entropy = np.mean(entropy_per_feature)
        return mean_entropy

    def adapt_adaptation_rate(self, mean_entropy):
        entropy_ratio = mean_entropy / 1.0
        min_rate = 0.1
        max_rate = 0.5
        self.adaptation_rate = min_rate + (max_rate - min_rate) * (1 - entropy_ratio)
        self.adaptation_rate = np.clip(self.adaptation_rate, min_rate, max_rate)

    def sample(self):
        sample = np.random.binomial(1, self.feature_probs).astype(bool)
        num_features = np.sum(sample)
        if num_features < self.min_features:
            indices = np.where(sample == 0)[0]
            np.random.shuffle(indices)
            sample[indices[:self.min_features - num_features]] = 1
        return sample

    def __call__(self, problem, n_samples, **kwargs):
        X = set()
        while len(X) < n_samples:
            sample = tuple(self.sample())
            X.add(sample)
        X = np.array(list(X))
        return Population.new("X", X)

    def update(self, population):
        if self.generation % self.update_frequency == 0:
            self.update_feature_importance()
        entropy = self.compute_entropy(population)
        self.adapt_adaptation_rate(entropy)
        self.adapt_probabilities(population)
        self.generation += 1

class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)
        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            _X[0, k] = np.logical_or(
                np.logical_and(p1, p2),
                np.logical_and(np.logical_xor(p1, p2), np.random.random(problem.n_var) < 0.5)
            )
        return _X

class AdaptiveMutation(Mutation):
    def __init__(self, adaptive_sampling):
        super().__init__()
        self.adaptive_sampling = adaptive_sampling

    def _do(self, problem, X, **kwargs):
        feature_probs = self.adaptive_sampling.feature_probs
        adaptation_rate = self.adaptive_sampling.adaptation_rate

        for i in range(len(X)):
            x = X[i, :]
            for j in range(len(x)):
                if x[j] == 1:
                    mutation_prob = (1 - feature_probs[j]) * adaptation_rate
                else:
                    mutation_prob = feature_probs[j] * adaptation_rate

                if np.random.rand() < mutation_prob:
                    x[j] = 1 - x[j]

            X[i, :] = x
        return X
    
class EarlyStoppingTermination(Termination):
    def __init__(self, max_generations, target_loss, **kwargs):
        super().__init__(**kwargs)
        self.max_generations = max_generations
        self.target_loss = target_loss
        
    def _update(self, algorithm):
        current_gen = getattr(algorithm, 'n_gen', 0)
        if current_gen is None:
            current_gen = 0

        progress_gen = min(1.0, current_gen / self.max_generations) if self.max_generations > 0 else 0.0
        if current_gen >= self.max_generations:
            print(f"Stop criteria max_generations = {current_gen}")
            return 1.0  
            
        try:
            valid_fitness = [ind.F[0] for ind in algorithm.pop if hasattr(ind, 'F') and ind.F is not None and len(ind.F) > 0 and ind.F[0] is not None] 
            if valid_fitness:
                best_fitness = min(valid_fitness)
                if best_fitness <= self.target_loss:
                    print(f"Stop criteria target_loss = {best_fitness:.5e} <= {self.target_loss:.5e}")
                    return 1.0 
        except Exception as e:
            print(f"Error checking fitness: {e}")     
        return progress_gen

class PLSProblem(ElementwiseProblem):
    def __init__(self, features: np.ndarray, target: np.ndarray, sss: StratifiedShuffleSplit, accuracy_scorer: make_scorer, adaptation_rate: float = 0.1, update_frequency: int = 3, min_features: int = 5, n_generations: int = 25, initial_top_k: float = 0.5, final_top_k: float = 0.1):
        super().__init__(n_var=features.shape[1], n_obj=1, n_constr=0, xl=np.zeros(features.shape[1]), xu=np.ones(features.shape[1]))
        self.features = features
        self.target = target
        self.sss = sss
        self.accuracy_scorer = accuracy_scorer
        self.adaptive_sampling = AdaptiveMySampling(
            features, target, sss, accuracy_scorer, FitnessEvaluator.find_optimal_n_components,
            adaptation_rate=adaptation_rate, update_frequency=update_frequency, min_features=min_features, n_generations=n_generations,
            initial_top_k=initial_top_k, final_top_k=final_top_k
        )

    def _evaluate(self, x, out, *args, **kwargs):
        selected_features = self.features[:, x == 1]
        if selected_features.shape[1] < 2:
          out["F"] = [1e6]
          return
        optimal_n_components = FitnessEvaluator.find_optimal_n_components(selected_features, self.target, self.sss, self.accuracy_scorer)
        error_cross_val = FitnessEvaluator.evaluate_pls_model(selected_features, self.target, optimal_n_components, self.sss, self.accuracy_scorer)
        out["F"] = [1 - error_cross_val]

    def update_sampling(self, population):
        self.adaptive_sampling.update(population)

class GeneticFeatureSelector:
    def __init__(self, X: np.ndarray, y: np.ndarray, population_size: int = 200, n_generations: int = 25, adaptation_rate: float = 0.1, update_frequency: int = 3, min_features: int = 5, initial_top_k: float = 0.5, final_top_k: float = 0.1):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.n_generations = n_generations
        self.adaptation_rate = adaptation_rate
        self.update_frequency = update_frequency
        self.min_features = min_features
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k
        self.sss = StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=42)
        self.accuracy_scorer = make_scorer(FitnessEvaluator.custom_accuracy)

    def select_features(self):
        problem = PLSProblem(self.X, self.y, sss=self.sss, accuracy_scorer=self.accuracy_scorer, adaptation_rate=self.adaptation_rate, update_frequency=self.update_frequency, min_features=self.min_features, n_generations=self.n_generations, initial_top_k=self.initial_top_k, final_top_k=self.final_top_k)
        adaptive_sampling = problem.adaptive_sampling
        algorithm = GA(
            pop_size=self.population_size,
            sampling=adaptive_sampling,
            crossover=BinaryCrossover(),
            mutation=AdaptiveMutation(adaptive_sampling),
            eliminate_duplicates=True
        )

        entropy_history = []

        def callback(algorithm):
            current_entropy = adaptive_sampling.compute_entropy(algorithm.pop)
            entropy_history.append(current_entropy)
            problem.update_sampling(algorithm.pop)

        termination = EarlyStoppingTermination(max_generations=self.n_generations, target_loss=1e-4)

        res = minimize(
            problem,
            algorithm,
            termination=termination,
            seed=1,
            verbose=False,
            save_history=True,
            callback=callback
        )
        return {
            "mask": res.X,
            "loss": res.F,
            "entropy_history": entropy_history,
            "algorithm_name": "VIP-GA"  
        }