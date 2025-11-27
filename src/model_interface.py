from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
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

        f1 = f1_score(y, y_pred)

        report_dict = classification_report(y, y_pred, output_dict=True)

        classes = list(report_dict.keys())[:-3]  # drop 'accuracy' 'macro avg' 'weighted avg'
        metrics = ["precision", "recall", "f1-score"]

        data = np.array([[report_dict[c][metric] for metric in metrics] for c in classes])

        fig, axis = plt.subplots(1, 3, figsize=(15, 4))
        sns.heatmap(data, annot=True, cmap="Greens", xticklabels=metrics, yticklabels=classes, fmt=".3f", ax=axis[0])
        axis[0].set_title("Classification Report (visualisation)")

        # CONFUSION MATRIX
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axis[1])
        axis[1].set_title("Matrice de confusion")
        axis[1].set_xlabel("Prédiction")
        axis[1].set_ylabel("Vérité")

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

        importances = np.array(importances)
        importances = np.maximum(importances, 0)  # clamp < 0 à 0
        importances /= importances.sum() + 1e-9  # normalisation

        # IMPORTANCES
        sns.barplot(x=np.arange(len(importances)), y=importances, ax=axis[2])
        axis[2].set_title("Importance des features (Permutation)")
        axis[2].set_xlabel("Features")
        axis[2].set_ylabel("Importance normalisée")
        axis[2].grid(axis="y")
        plt.show()

    def explain(self, X_train: np.ndarray, X_test: np.ndarray, feature_names: list, **kwargs):
        raise NotImplementedError(f"La méthode explain() n'est pas implémentée pour {self.__class__.__name__}")
