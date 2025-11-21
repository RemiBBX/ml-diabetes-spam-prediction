from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from numpy import ravel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


class KNNModel(LearningModelInterface):
    K = 3

    def __init__(self) -> None:
        self.model = KNeighborsClassifier(n_neighbors=self.K)

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, ravel(y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(x)
        return prediction


class MLP_nn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_size=3, drop_rate=0.4):
        super(MLP_nn, self).__init__()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_rate)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(layer_size - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)

        return x


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    print_every_epochs=1,
):
    best_val_loss = float("inf")
    best_state = None

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer,
    #     T_max=epochs,
    #     eta_min=1e-6
    # )

    train_losses, valid_losses = [], []

    for i in range(epochs):
        ## Training
        model.train()
        train_loss, valid_loss = 0, 0
        for data, label in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
        # scheduler.step()

        ## Validation against val loader
        model.eval()
        for data, label in val_loader:
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.sampler)
        valid_loss /= len(val_loader.sampler)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if i % print_every_epochs == 0:
            print(
                "epoch: {} \ttraining Loss: {:.6f} ".format(
                    i + 1,
                    train_loss,
                )
            )
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_state = model.state_dict()
            print(f"Validation loss improved → new best = {best_val_loss:.6f}")
        else:
            print(
                f"No improvement in validation loss (current: {valid_loss:.6f}, best: {best_val_loss:.6f}) "
                f"→ reverting weights"
            )
            model.load_state_dict(best_state)

    return train_losses, valid_losses


class MLPModel(LearningModelInterface):
    def __init__(self, input_size: Literal[57, 21], epochs) -> None:
        ## Parameters
        self.input_size = input_size
        self.layer_size = 3
        self.hidden_size = 128

        self.train_size = 0.7
        self.validation_size = 0.15
        self.test_size = 0.15

        self.learning_rate = 0.001
        self.batch_size = 512
        self.epochs = epochs

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


data_diabetes = "./data/diabetes_binary_health_indicators_BRFSS2015.csv"
data_spam = "./data/spambase/spambase.data"

# Récupération des features de spambase
column_names = []
with open("./data/spambase/spambase.names", "r") as f:
    for line in f:
        line = line.strip()
        # ignorer commentaires, lignes vides, et la ligne de classes
        if not line or line.startswith("|") or line.startswith("1,"):
            continue
        # ne garder que les lignes "attribut: type."
        if ":" in line:
            col = line.split(":")[0].strip()
            if col.lower() == "class":
                continue
            column_names.append(col)
column_names.append("Class")

train_size = 0.7
validation_size = 0.15
test_size = 0.15


@dataclass
class PreprocessedData:
    X_train: np.ndarray
    X_test: np.ndarray
    X_validation: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_validation: np.ndarray


def load_data(data):
    df = pd.read_csv(data)
    df = df.fillna(df.median())  # gérer les valeurs manquantes
    df = df.sample(frac=1, random_state=42)  # réduit le nombre de ligne pour des tests

    if data == data_diabetes:
        df = df.rename(columns={"Diabetes_binary": "Class"})
    if data == data_spam:
        df.columns = column_names
    y = df["Class"]
    X = df.drop(columns=["Class"])

    feature_names = list(df.columns)

    return X, y, df, feature_names


def detect_binary_and_continuous(df):
    binary_cols = []
    continuous_cols = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:  # binaire
            binary_cols.append(col)
        else:
            continuous_cols.append(col)
    return binary_cols, continuous_cols


def preprocessing(data, test_size, validation_size):
    X, y, df, _ = load_data(data)
    _, continuous_cols = detect_binary_and_continuous(df)

    ## Standardisation avec scikit, seulement les features continues
    scaler = StandardScaler()
    X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
    y = y.to_numpy().reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + validation_size,
        random_state=42,
        stratify=y,  # Classes en même proportion que dans le dataset entier
    )
    if validation_size != 0:
        X_test, X_validation, y_test, y_validation = train_test_split(
            X_temp,
            y_temp,
            test_size=validation_size / (test_size + validation_size),
            random_state=42,
            stratify=y_temp,  # classes en même proportion que dans le dataset entier
        )
    else:
        X_test, X_validation, y_test, y_validation = X_temp, [], y_temp, []

    return PreprocessedData(
        X_train=X_train.to_numpy(),
        X_test=X_test.to_numpy(),
        X_validation=X_validation.to_numpy(),
        y_train=y_train,
        y_test=y_test,
        y_validation=y_validation,
    )


def visualize(data, selected_features, random=False):
    X, y, df, feature_names = load_data(data)
    _, continuous_cols = detect_binary_and_continuous(df)

    ## Standardisation avec scikit, seulement les features continues
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    feature_names.remove("Class")
    # np.random.seed(42)  # pour reproductibilité
    if random:
        selected_features = np.random.choice(feature_names, size=5, replace=True)
    selected_features = list(selected_features)

    df["Class"] = df["Class"].to_numpy()
    if "Class" not in selected_features:
        selected_features += ["Class"]
    sns.pairplot(df[selected_features], hue="Class")
    plt.show()

    # Matrice de corrélation
    corr_matrix = df[feature_names].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Coefficient de Corrélation"},
    )
    plt.title("Heatmap de la Matrice de Corrélation")
    plt.show()


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
