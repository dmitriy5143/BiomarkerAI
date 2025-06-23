import numpy as np
from numba import njit
from pipeline.fitness_evaluator import FitnessEvaluator
from sklearn.metrics import accuracy_score, roc_curve, make_scorer
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import check_random_state
from joblib import parallel_backend
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

@njit
def fisher_score(X, y):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    scores = np.zeros(n_features)

    for j in range(n_features):
        numerator = 0
        feature = X[:, j]
        overall_mean = np.mean(feature)
        overall_variance = np.var(feature)
        for c in classes:
            class_samples = feature[y == c]
            n_c = len(class_samples)
            class_mean = np.mean(class_samples)
            numerator += n_c * (class_mean - overall_mean) ** 2
        scores[j] = numerator / (overall_variance + 1e-6)
    return scores

def mic_score(X, y):
    scores = mutual_info_classif(X, y, random_state=42)
    return scores

def remove_duplicates(population):
    unique = []
    chromosomes_set = set()
    for individual in population:
        chr_str = ''.join(map(str, individual['chromosome']))
        if chr_str not in chromosomes_set:
            chromosomes_set.add(chr_str)
            unique.append(individual)
    return unique

def fast_non_dominated_sort(population):
    objectives = np.array([ind['objectives'] for ind in population])
    nds = NonDominatedSorting().do(objectives)
    fronts = []
    for rank in range(len(nds)):
        front = [population[i] for i in nds[rank]]
        for ind in front:
            ind['rank'] = rank
        fronts.append(front)
    return fronts

def dominates(p, q):
    and_condition = all(p_o <= q_o for p_o, q_o in zip(p['objectives'], q['objectives']))
    or_condition = any(p_o < q_o for p_o, q_o in zip(p['objectives'], q['objectives']))
    return and_condition and or_condition

def diversity_based_environmental_selection(fronts, N):
    new_population = []
    for front in fronts:
        if len(new_population) + len(front) <= N:
            new_population.extend(front)
        else:
            ds_values = calculate_ds(front, new_population)
            sorted_front = [x for _, x in sorted(zip(ds_values, front), key=lambda pair: pair[0], reverse=True)]
            remaining = N - len(new_population)
            new_population.extend(sorted_front[:remaining])
            break
    return new_population

def calculate_ds(front, existing_population):
    T = existing_population + front
    chromosomes_T = np.array([x['chromosome'] for x in T])         
    chromosomes_front = np.array([xi['chromosome'] for xi in front]) 
    cosine_similarities = cosine_similarity(chromosomes_front, chromosomes_T)  
    ids_front = np.array([id(xi) for xi in front])  
    ids_T = np.array([id(xj) for xj in T])          
    match_matrix = ids_front[:, np.newaxis] == ids_T[np.newaxis, :]  
    cosine_similarities[match_matrix] = -np.inf
    max_cosines = np.max(cosine_similarities, axis=1)  
    ds_values = 1 - max_cosines
    return ds_values

def nondominated_solution(candidates):
    nondominated = []
    for p in candidates:
        dominated = False
        for q in candidates:
            if dominates(q, p):
                dominated = True
                break
        if not dominated:
            nondominated.append(p)
    if len(nondominated) == 0:
        return candidates[np.random.randint(len(candidates))]
    else:
        return nondominated[np.random.randint(len(nondominated))]

class DMBDE:
    def __init__(self, X, y, population_size=100, n_generations=100, a=0.1, sigma=0.005, CR=0.5, F=0.5, random_state=None):
        self.X = X
        self.y = y
        self.N = population_size # Размер популяции
        self.n_generations = n_generations  # Максимальное число итераций (поколений)
        self.a = a                # Коэффициент для вычисления δ
        self.sigma = sigma        # Коэффициент турбулентности
        self.CR = CR              # Параметр кроссовера
        self.F = F                # Параметр мутации 
        self.sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        self.accuracy_scorer = make_scorer(FitnessEvaluator.custom_accuracy)
        self.random_state = check_random_state(random_state)
        self.preprocess_features()

    def preprocess_features(self):
        fisher_scores = fisher_score(self.X, self.y)
        mic_scores = mic_score(self.X, self.y)
        scaler = MinMaxScaler()
        fisher_scores_norm = scaler.fit_transform(fisher_scores.reshape(-1, 1)).flatten()
        mic_scores_norm = scaler.fit_transform(mic_scores.reshape(-1, 1)).flatten()
        scores = 0.5 * fisher_scores_norm + 0.5 * mic_scores_norm
        delta = self.a * np.max(scores)
        self.selected_features_indices = np.where(scores > delta)[0]
        if len(self.selected_features_indices) == 0:
            self.selected_features_indices = np.array([np.argmax(scores)])

        self.X_selected = self.X[:, self.selected_features_indices]
        self.n_features_selected = self.X_selected.shape[1]

    def evaluate_individual(self, individual):
        selected_indices = np.where(individual == 1)[0]
        if len(selected_indices) == 0:
            return 1.0, 0.0 
        X_sub = self.X_selected[:, selected_indices]
        n_components = FitnessEvaluator.find_optimal_n_components(X_sub, self.y, self.sss, self.accuracy_scorer)
        accuracy = FitnessEvaluator.evaluate_pls_model(X_sub, self.y, n_components, self.sss, self.accuracy_scorer)
        feature_ratio = len(selected_indices) / self.n_features_selected
        return 1 - accuracy, feature_ratio

    def algorithm2(self, population):
        N = len(population)
        O = []
        objectives = np.array([ind['objectives'] for ind in population])
        scaler = MinMaxScaler() 
        normalized_objectives = scaler.fit_transform(objectives)
        distances = squareform(pdist(normalized_objectives))
        Nei = []

        for i in range(N):
            sorted_indices = np.argsort(distances[i])
            Nei_i = sorted_indices[1:4].tolist()
            mu_i = np.mean(distances[i, Nei_i])
            sigma_i = np.std(distances[i, Nei_i])
            lower_bound = mu_i - 3 * sigma_i
            upper_bound = mu_i + 3 * sigma_i
            for j in range(N):
                if j != i and lower_bound <= distances[i, j] <= upper_bound:
                    if j not in Nei_i:
                        Nei_i.append(j)
            Nei.append(Nei_i)

        for i in range(N):
            if np.random.rand() < 0.8 and len(Nei[i]) >= 3:
                parents_indices = np.random.choice(Nei[i], 3, replace=False)
            else:
                parents_indices = np.random.choice(N, 3, replace=False)
            xr1 = population[parents_indices[0]]['chromosome']
            xr2 = population[parents_indices[1]]['chromosome']
            xr3 = population[parents_indices[2]]['chromosome']
            candidates = [population[parents_indices[0]], population[parents_indices[1]], population[parents_indices[2]]]
            xlbest = nondominated_solution(candidates)
            xi = population[i]['chromosome']

            vi = np.zeros_like(xi)
            for j in range(len(xi)):
                if dominates(xlbest, population[i]):
                    Ci_j = self.sigma
                else:
                    diff = xr1[j] ^ xr2[j]
                    Ci_j = min(1, diff + self.sigma)
                if Ci_j < np.random.rand():
                    vi[j] = xlbest['chromosome'][j]
                else:
                    vi[j] = 1 - xlbest['chromosome'][j]

            ui = np.zeros_like(vi)
            j_rand = np.random.randint(len(vi))
            for j in range(len(vi)):
                if np.random.rand() <= self.CR or j == j_rand:
                    ui[j] = vi[j]
                else:
                    ui[j] = xi[j]

            error, feature_ratio = self.evaluate_individual(ui)
            O.append({'chromosome': ui, 'objectives': [error, feature_ratio]})
        return O

    def has_converged(self, population, tol=1e-4):
        objectives = np.array([ind['objectives'] for ind in population])
        min_error = np.min(objectives[:, 0])
        if min_error <= tol:
            print('Early stopping triggered! Minimum error reached:', min_error)
            return True
        return False

    def select_features(self):
        P = self.random_state.randint(0, 2, size=(self.N, self.n_features_selected))
        population = []
        for individual in P:
            error, feature_ratio = self.evaluate_individual(individual)
            population.append({'chromosome': individual.copy(), 'objectives': [error, feature_ratio]})
        
        for generation in range(1, self.n_generations + 1):
            offspring = self.algorithm2(population)
            combined_population = population + offspring
            combined_population = remove_duplicates(combined_population)
            fronts = fast_non_dominated_sort(combined_population)
            population = diversity_based_environmental_selection(fronts, self.N)
            
            current_objectives = np.array([ind['objectives'] for ind in population])
            best_error = np.min(current_objectives[:, 0])
            mean_error = np.mean(current_objectives[:, 0])
            mean_ratio = np.mean(current_objectives[:, 1])
            print(f"\nПоколение {generation}:")
            print(f"Лучшая ошибка: {best_error:.4f}")
            print(f"Средняя ошибка: {mean_error:.4f}")
            print(f"Среднее отношение признаков: {mean_ratio:.4f}")

            if self.has_converged(population):
                break

        final_fronts = fast_non_dominated_sort(population)
        final_front = final_fronts[0]
        final_feature_subsets = []
        for individual in final_front:
            selected_indices = np.where(individual['chromosome'] == 1)[0]
            features = self.selected_features_indices[selected_indices]
            final_feature_subsets.append(features)
        sorted_solutions = sorted(final_front, key=lambda x: x['objectives'][0])
        best_solution = sorted_solutions[0]
        selected_indices = np.where(best_solution['chromosome'] == 1)[0]
        best_features = self.selected_features_indices[selected_indices]
        final_loss = best_solution['objectives'][0]
        return {
            "mask": best_features,
            "loss": final_loss,
            "entropy_history": None,  
            "algorithm_name": "DMBDE" 
        }