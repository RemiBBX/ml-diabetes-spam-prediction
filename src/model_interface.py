from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class LearningModelInterface(ABC):
    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def benchmark(self, x: np.ndarray, y: np.ndarray):
        y_pred = self.predict(x)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        print("\n===== BENCHMARK DU MODÈLE =====")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-score  : {f1:.4f}")

        print("\n===== Classification Report =====")
        report_dict = classification_report(y, y_pred, output_dict=True)
        print(classification_report(y, y_pred))

        classes = list(report_dict.keys())[:-3]  # drop 'accuracy' 'macro avg' 'weighted avg'
        metrics = ["precision", "recall", "f1-score"]

        data = np.array([[report_dict[c][metric] for metric in metrics] for c in classes])

        plt.figure(figsize=(7, 3))
        sns.heatmap(data, annot=True, cmap="Greens", xticklabels=metrics, yticklabels=classes, fmt=".3f")
        plt.title("Classification Report (visualisation)")
        plt.show()

        # CONFUSION MATRIX
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matrice de confusion")
        plt.xlabel("Prédiction")
        plt.ylabel("Vérité")
        plt.show()

        print("\n===== IMPORTANCE DES FEATURES =====")

        n_features = x.shape[1]
        importances = []

        baseline_score = f1

        for i in range(n_features):
            x_permuted = x.copy()

            np.random.shuffle(x_permuted[:, i])

            yp = self.predict(x_permuted)
            if yp.ndim > 1:
                yp = (yp[:, 0] > 0.5).astype(int)

            perm_score = f1_score(y, yp)

            importance = baseline_score - perm_score
            importances.append(importance)

            print(f"Feature {i} : importance = {importance:.4f}")

        importances = np.array(importances)
        importances = np.maximum(importances, 0)  # clamp < 0 à 0
        importances /= importances.sum() + 1e-9  # normalisation

        # IMPORTANCES
        plt.figure(figsize=(7, 4))
        sns.barplot(x=np.arange(len(importances)), y=importances)
        plt.title("Importance des features (Permutation)")
        plt.xlabel("Features")
        plt.ylabel("Importance normalisée")
        plt.grid(axis="y")
        plt.show()
