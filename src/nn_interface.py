from typing import Literal

import numpy as np
import torch.nn
from torch.utils.data import DataLoader, TensorDataset

from src.model_interface import LearningModelInterface
from src.neural_network import MLP_nn, train_model


class MLPModel(LearningModelInterface):
    def __init__(self, input_size: Literal[57, 21]) -> None:
        ## Parameters
        self.input_size = input_size
        self.layer_size = 3
        self.hidden_size = 128

        self.train_size = 0.7
        self.validation_size = 0.15
        self.test_size = 0.15

        self.learning_rate = 0.001
        self.batch_size = 512
        self.epochs = 15

        self.model = MLP_nn(
            input_size=self.input_size, hidden_size=self.hidden_size, output_size=1, layer_size=self.layer_size
        )

    def train(self, samples) -> None:
        self.X_train = torch.tensor(samples.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(samples.y_train, dtype=torch.float32)
        self.X_val = torch.tensor(samples.X_validation, dtype=torch.float32)
        self.y_val = torch.tensor(samples.y_validation, dtype=torch.float32)
        self.X_test = torch.tensor(samples.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(samples.y_test, dtype=torch.float32)

        self.train_loader = DataLoader(
            TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.validation_loader = DataLoader(
            TensorDataset(self.X_val, self.y_val),
            batch_size=self.batch_size,
            shuffle=False,
        )
        pos = torch.count_nonzero(self.y_train)
        pos_weight = torch.tensor([len(self.y_train) / pos], dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=self.epochs,
            print_every_epochs=1,
        )

    def predict(self, x) -> np.ndarray:
        X_test = torch.tensor(x, dtype=torch.float32)
        probs = torch.sigmoid(self.model(X_test)).detach().cpu().numpy().flatten()

        t = 0.5
        pred = (probs >= t).astype(int)
        return pred
