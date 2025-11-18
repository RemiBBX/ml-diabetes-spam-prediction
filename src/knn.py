import numpy as np
from numpy.ma.core import ravel
from sklearn.neighbors import KNeighborsClassifier

from src.model_interface import LearningModelInterface


class KNNModel(LearningModelInterface):
    def __init__(self, k) -> None:
        self.model = KNeighborsClassifier(n_neighbors=k)

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, ravel(y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(x)
        return prediction
