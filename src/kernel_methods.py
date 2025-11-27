import numpy as np
from numpy.ma.core import ravel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from src.model_interface import LearningModelInterface


class LinearSVC_(LearningModelInterface):
    def __init__(self) -> None:
        self.model = None
        self.best_params_ = None
        self.param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        grid = GridSearchCV(
            LinearSVC(class_weight="balanced", max_iter=5000), param_grid=self.param_grid, cv=5, scoring="f1_macro"
        )
        grid.fit(x, ravel(y))
        self.best_params_ = grid.best_params_
        self.model = grid.best_estimator_
        print("Best hyperparameters found with Grid Search:", self.best_params_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)
