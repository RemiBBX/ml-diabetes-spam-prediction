import torch.nn
import numpy as np
from fontTools.misc.classifyTools import classify
from torch.utils.data import DataLoader, TensorDataset

from src.model_interface import LearningModelInterface
from src.neural_network import MLP_nn, train_model
from src.preprocessing import preprocessing, data_diabetes, data_spam
from src.utils import compute_accuracy
from sklearn.metrics import confusion_matrix, classification_report

class MLPModel(LearningModelInterface):
    def __init__(self) -> None:
        ## Parameters
        self.input_size = 57
        self.layer_size = 3
        self.hidden_size = 128

        self.train_size = 0.7
        self.validation_size = 0.15
        self.test_size = 0.15

        self.learning_rate = 0.001
        self.epochs = 10

        self.model = MLP_nn(input_size=self.input_size, hidden_size=self.hidden_size, output_size=1, layer_size=self.layer_size)

        dataset = preprocessing(
            data=data_spam,
            test_size=self.test_size,
            validation_size=self.validation_size,
        )
        X_train = torch.tensor(dataset.X_train, dtype=torch.float32)
        y_train = torch.tensor(dataset.y_train, dtype=torch.float32)
        X_test = torch.tensor(dataset.X_test, dtype=torch.float32)
        y_test = torch.tensor(dataset.y_test, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=512,
            shuffle=True,
        )
        X_val = torch.tensor(dataset.X_validation, dtype=torch.float32)
        y_val = torch.tensor(dataset.y_validation, dtype=torch.float32)

        validation_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=512,
            shuffle=False,
        )

    def train(self) -> None:
        pos = torch.count_nonzero(self.y_train)
        pos_weight = torch.tensor([len(self.y_train) / pos], dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_model(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.validation_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=self.epochs,
            print_every_epochs=1
        )

    def predict(self) -> np.ndarray:
        probs = torch.sigmoid(self.model(self.X_test)).detach().cpu().numpy().flatten()
        y_true = self.y_test.numpy().flatten()

        t = 0.5
        pred = (probs >= t).astype(int)
        return pred
