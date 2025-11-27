from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_diabetes = "./data/diabetes_binary_health_indicators_BRFSS2015.csv"
data_spam = "./data/spambase/spambase.data"
# data_spam = "/Users/alexis/Documents/Scolarité/REPO/projet-ml/data/spambase/spambase.data"

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


if __name__ == "__main__":
    visualize(data_diabetes, [], random=True)
