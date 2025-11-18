from abc import ABC, abstractmethod

import numpy as np


class LearningModelInterface(ABC):
    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def benchmark(self):
        print()
