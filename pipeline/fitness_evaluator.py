import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from joblib import parallel_backend

class FitnessEvaluator:
    @staticmethod
    def evaluate_pls_model(features, target, n_components, sss, accuracy_scorer):
        pls = PLSRegression(n_components=n_components, scale=True)
        with parallel_backend('loky', n_jobs=-1):
            scores = cross_val_score(pls, features, target, cv=sss, scoring=accuracy_scorer)
        return scores.mean()

    @staticmethod
    def find_optimal_n_components(X, y, sss, accuracy_scorer, max_components=4):
        n_features = X.shape[1]
        if n_features == 0:
          return 1
        max_components = min(max_components, n_features)
        accuracy_cv = []
        for n_comp in range(1, max_components + 1):
            score = FitnessEvaluator.evaluate_pls_model(X, y, n_comp, sss, accuracy_scorer)
            accuracy_cv.append(score)
        optimal_n_components = np.argmax(accuracy_cv) + 1
        return optimal_n_components

    @staticmethod
    def find_optimal_threshold(y_true, y_pred_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        accuracies = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            accuracies.append(accuracy_score(y_true, y_pred))
        optimal_idx = np.argmax(accuracies)
        return thresholds[optimal_idx]

    @staticmethod
    def custom_accuracy(y_true, y_pred):
        optimal_threshold = FitnessEvaluator.find_optimal_threshold(y_true, y_pred)
        y_pred_class = (y_pred >= optimal_threshold).astype(int)
        return accuracy_score(y_true, y_pred_class)