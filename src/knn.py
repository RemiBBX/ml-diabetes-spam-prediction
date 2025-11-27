import joblib
import numpy as np
from numpy.ma.core import ravel
from sklearn.neighbors import KNeighborsClassifier

from src.model_interface import LearningModelInterface


class KNNModel(LearningModelInterface):
    K = 3

    def __init__(self) -> None:
        self.model = KNeighborsClassifier(n_neighbors=self.K)

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, ravel(y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(x)
        return prediction

    def save_model(self, path: str):
        """Sauvegarde le modèle entraîné (grid.best_estimator_)"""
        if self.model is not None:
            joblib.dump(self.model, path)
            print(f" Modèle KNN sauvegardé dans {path}")
        else:
            print(" Erreur: Le modèle n'a pas été entraîné. Veuillez appeler train() d'abord.")

    def load_model(self, path: str):
        """Charge un modèle entraîné depuis un fichier"""
        try:
            self.model = joblib.load(path)
            # Récupérer les paramètres pour référence si nécessaire
            self.best_params_ = self.model.get_params()
            print(f" Modèle KNN chargé depuis {path}")
            return True
        except FileNotFoundError:
            print(f" Erreur: Fichier non trouvé à {path}")
            return False
        except Exception as e:
            print(f" Erreur lors du chargement: {e}")
            return False
