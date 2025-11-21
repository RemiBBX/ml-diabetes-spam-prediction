import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.model_interface import LearningModelInterface


class RForest(LearningModelInterface):
    def __init__(self):
        self.model = None
        self.best_params_ = None

        self.param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_leaf": [
                1,
                2,
            ],
        }

    def train(self, x, y):
        grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state=0), param_grid=self.param_grid, cv=5, n_jobs=1, verbose=0
        )

        grid.fit(x, np.ravel(y))

        self.best_params_ = grid.best_params_
        self.model = grid.best_estimator_
        print("Best hyperparameters found with Grid Search:", self.best_params_)

    def predict(self, x):
        return self.model.predict(x)
