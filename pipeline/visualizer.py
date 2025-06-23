import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import make_scorer
from pipeline.fitness_evaluator import FitnessEvaluator
import os
import numpy as np
import pandas as pd
import time

class Visualizer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.sss = StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=42)
        self.accuracy_scorer = make_scorer(FitnessEvaluator.custom_accuracy)
        self.n_components = FitnessEvaluator.find_optimal_n_components(self.X, self.y, self.sss, self.accuracy_scorer)
        self.model = PLSRegression(n_components=self.n_components, scale=True)
        self.model.fit(self.X, self.y)
        
    def plot_predictions(self, save_path="static/plots"):
        os.makedirs(save_path, exist_ok=True)
        y_pred = self.model.predict(self.X).ravel()
        plt.figure(figsize=(10, 8))
        colors = np.where(np.array(self.y).ravel() == 0, 'blue', 'red')
        plt.scatter(self.y, y_pred, alpha=0.5, edgecolor='k', c=colors, label='Класс 0 и 1')
        plt.axhline(y=0.5, color='green', linestyle='--', label='Граница принятия решений')
        plt.plot([0, 1], [0, 1], color='black', linestyle='-')
        plt.xlabel('Истинные значения')
        plt.ylabel('Предсказанные значения')
        plt.grid(True)
        plt.tight_layout()
        
        filename = f"predictions_{int(time.time())}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath)
        plt.close()
        
        return {
            "url": f"/static/plots/{filename}",
            "title": "Предсказания модели"
        }
        
    def plot_coefficients(self, save_path="static/plots"):
        os.makedirs(save_path, exist_ok=True)
        coefficients = self.model.coef_.ravel()
        if isinstance(self.X, pd.DataFrame):
            feature_names = self.X.columns.tolist()
        else:
            feature_names = [f'Фича {i+1}' for i in range(len(coefficients))]
            
        plt.figure(figsize=(10, 8))
        bars = plt.bar(range(len(coefficients)), coefficients, color='red', alpha=0.7, edgecolor='black', linewidth=3)
        plt.xlabel('Индексы выбранных переменных', fontsize=14, fontweight='bold')
        plt.ylabel('Значения регрессионных коэффициентов', fontsize=14, fontweight='bold')
        plt.xticks(range(len(coefficients)), feature_names, rotation=90, ha='right', fontsize=15)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}',
                    ha='center', va='bottom', rotation=90, fontsize=10, fontweight='bold')
                    
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = f"coefficients_{int(time.time())}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath)
        plt.close()
        
        return {
            "url": f"/static/plots/{filename}",
            "title": "Коэффициенты модели"
        }
    
    def plot_entropy_evolution(self, entropy_history, save_path="static/plots"):
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(10, 8))
        plt.plot(range(len(entropy_history)), entropy_history, marker='o', linestyle='-', color='purple')
        plt.title("Эволюция энтропии популяции")
        plt.xlabel("Поколение")
        plt.ylabel("Средняя энтропия")
        plt.grid(True)
        plt.tight_layout()
        filename = f"entropy_evolution_{int(time.time())}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath)
        plt.close()
        return {
            "url": f"/static/plots/{filename}",
            "title": "Эволюция Энтропии Популяции"
        }