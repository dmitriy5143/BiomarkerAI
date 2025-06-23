import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state
from pipeline.fitness_evaluator import FitnessEvaluator

def logistic_map(size, dim, iterations=100, r=4.0):
    x = np.random.rand(size, dim)
    for _ in range(iterations):
        x = r * x * (1 - x)
    x = 2 * x - 1
    return x

def coefficient_functions(x):
    f1 = (2 / 3) * np.tanh(2 * x - 0.5)
    f2 = np.sin(np.cos(2 * np.pi * x ** 2))
    f3 = np.cos(np.sin((np.pi / 2) * x ** 2))
    f4 = np.arccos(np.cos((np.pi / 4) * x ** 2))
    return [f1, f2, f3, f4]

class PSOVA2:
    def __init__(self,
                 X=None,
                 y=None,
                 population_size=30,   
                 n_generations=100,    
                 alpha=0.9,
                 beta=0.1,
                 early_stopping_patience=10,     
                 early_stopping_threshold=1e-4,
                 random_state=None):
        self.n_particles = population_size
        self.n_dimensions = X.shape[1]
        self.max_iter = n_generations
        self.alpha = alpha   
        self.beta = beta
        self.X = X
        self.y = y
        self.sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        self.accuracy_scorer = make_scorer(FitnessEvaluator.custom_accuracy)
        self.random_state = check_random_state(random_state)
        self.x_min = -1
        self.x_max = 1
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.no_improvement_count = 0
        self.best_fitness_history = []  
        self.initialize_swarm()

    def initialize_swarm(self):
        self.positions = logistic_map(self.n_particles, self.n_dimensions)
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(self.n_particles, -np.inf)
        self.gbest_position = None
        self.gbest_score = -np.inf
        self.evaluate_swarm()

    def evaluate_function(self, features):
        X_selected = self.X[:, features]
        if X_selected.shape[1] == 0:
            return -np.inf
        else:
            n_components = FitnessEvaluator.find_optimal_n_components(X_selected, self.y, self.sss, self.accuracy_scorer)
            accuracy = FitnessEvaluator.evaluate_pls_model(X_selected, self.y, n_components, self.sss, self.accuracy_scorer)
            return accuracy

    def evaluate_swarm(self):
        for i in range(self.n_particles):
            features = self.positions[i] > 0
            num_selected = np.sum(features)
            if num_selected == 0:
                fitness = -np.inf
            else:
                accuracy = self.evaluate_function(features)
                fitness = self.alpha * accuracy + self.beta * (1 / num_selected)
            if fitness > self.pbest_scores[i]:
                self.pbest_scores[i] = fitness
                self.pbest_positions[i] = self.positions[i]
            if fitness > self.gbest_score:
                self.gbest_score = fitness
                self.gbest_position = self.positions[i]

    def swarm_leader_enhancement(self):
        x = np.random.rand()
        g = 2 / (1 + np.exp(1 - 2 * x)) - 1
        delta = g * (self.x_max - self.x_min)
        new_position = self.gbest_position + delta
        new_position = np.clip(new_position, self.x_min, self.x_max)
        features = new_position > 0
        num_selected = np.sum(features)
        if num_selected == 0:
            fitness = -np.inf
        else:
            accuracy = self.evaluate_function(features)
            fitness = self.alpha * accuracy + self.beta * (1 / num_selected)
        if fitness > self.gbest_score:
            self.gbest_score = fitness
            self.gbest_position = new_position

    def worst_solution_enhancement(self):
        worst_indices = np.argsort(self.pbest_scores)[:3]
        for idx in worst_indices:
            beta = np.random.rand(self.n_dimensions)
            x_donor1d = self.x_min + beta * (self.x_max - self.x_min)
            pbest_random = self.pbest_positions[np.random.choice(self.n_particles)]
            x_donor2d = pbest_random
            if np.random.rand() <= 0.5:
                new_position = x_donor1d
            else:
                new_position = x_donor2d
            new_position = np.clip(new_position, self.x_min, self.x_max)

            features = new_position > 0
            num_selected = np.sum(features)
            if num_selected == 0:
                fitness = -np.inf
            else:
                accuracy = self.evaluate_function(features)
                fitness = self.alpha * accuracy + self.beta * (1 / num_selected)

            if fitness > self.pbest_scores[idx]:
                self.pbest_scores[idx] = fitness
                self.pbest_positions[idx] = new_position
                self.positions[idx] = new_position

    def construct_breeding_exemplar(self, iteration):
        iter_ratio = iteration / self.max_iter

        if iteration <= 0.25 * self.max_iter:
            c1, c2, c3 = np.random.rand(3)
            selected_indices = np.random.choice(self.n_particles, 3, replace=False)
            numerator = c1 * self.pbest_positions[selected_indices[0]] \
                      + c2 * self.pbest_positions[selected_indices[1]] \
                      + c3 * self.pbest_positions[selected_indices[2]]
            denominator = c1 + c2 + c3
            x_offspring = numerator / denominator
        elif iteration <= 0.5 * self.max_iter:
            c1, c2 = np.random.rand(2)
            selected_indices = np.random.choice(self.n_particles, 2, replace=False)
            numerator = c1 * self.pbest_positions[selected_indices[0]] \
                      + c2 * self.pbest_positions[selected_indices[1]]
            denominator = c1 + c2
            x_offspring = numerator / denominator
        elif iteration <= 0.75 * self.max_iter:
            selected_index = np.random.choice(self.n_particles)
            x_offspring = self.pbest_positions[selected_index]
        else:
            x_offspring = np.zeros(self.n_dimensions)

        m1 = 0.4 + 0.5 * np.sin(np.pi / 2 * iter_ratio) * np.sinh(iter_ratio)
        m2 = 0.4 * np.cos(np.pi / 2 * iter_ratio) * np.cosh(iter_ratio)

        x_d_exemplar = m1 * self.gbest_position + m2 * x_offspring
        x_d_exemplar = np.clip(x_d_exemplar, self.x_min, self.x_max)

        return x_d_exemplar

    def update_particles(self, iteration):
        new_positions = np.copy(self.positions)
        x_d_exemplar = self.construct_breeding_exemplar(iteration)

        for i in range(self.n_particles):
            phi_funcs = coefficient_functions(np.random.rand(self.n_dimensions))
            phi = phi_funcs[np.random.choice(len(phi_funcs))]
            gaussian_noise = np.random.normal(0, 0.01, self.n_dimensions)
            for j in range(self.n_dimensions):
                if np.random.rand() < 0.4:
                    target = x_d_exemplar[j]
                else:
                    target = self.gbest_position[j]

                new_positions[i, j] = self.positions[i, j] + phi[j] * (target - self.positions[i, j]) + gaussian_noise[j]
            new_positions[i] = np.clip(new_positions[i], self.x_min, self.x_max)
        self.positions = new_positions

    def check_early_stopping(self, current_fitness):
        if not self.best_fitness_history:
            self.best_fitness_history.append(current_fitness)
            return False
        last_best = self.best_fitness_history[-1]
        improvement = current_fitness - last_best
        if improvement <= self.early_stopping_threshold:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        self.best_fitness_history.append(current_fitness)
        if self.no_improvement_count >= self.early_stopping_patience:
            print(f"Early stopping triggered! No significant improvement for {self.early_stopping_patience} iterations.")
            return True
        return False

    def optimize(self):
        for iteration in range(1, self.max_iter + 1):
            self.swarm_leader_enhancement()
            self.worst_solution_enhancement()
            self.update_particles(iteration)
            self.evaluate_swarm()
            
            num_selected_features = np.sum(self.gbest_position > 0)
            print(f"Iteration {iteration}/{self.max_iter}, Best Fitness: {self.gbest_score:.4f}, Selected Features: {num_selected_features}")
            
            if iteration > 1 and self.check_early_stopping(self.gbest_score):
                print(f"Early stopping at iteration {iteration}/{self.max_iter}")
                break
                    
        return self.gbest_position, self.gbest_score

    def select_features(self):
        best_position, best_fitness = self.optimize()
        feature_mask = best_position > 0
        return {
            "mask": feature_mask,
            "loss": best_fitness,
            "entropy_history": None,
            "algorithm_name": "PSOVA2"
        }