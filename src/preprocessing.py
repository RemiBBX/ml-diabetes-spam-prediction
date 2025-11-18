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

column_names = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
    "Class",
]

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


def preprocessing(data, test_size, validation_size):
    X, y, df, _ = load_data(data)

    ## Standardisation avec scikit
    X = StandardScaler().fit_transform(X)  # moyenne nulle et variance unité
    y = y.to_numpy().reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size + validation_size,
        random_state=42,
        stratify=y,  # Classes en même proportion que dans le dataset entier
    )
    if validation_size != 0 :
        X_test, X_validation, y_test, y_validation = train_test_split(
            X_temp,
            y_temp,
            test_size=validation_size / (test_size + validation_size),
            random_state=42,
            stratify=y_temp,  # classes en même proportion que dans le dataset entier
        )
    else :
        X_test, X_validation, y_test, y_validation = X_temp, [], y_temp, []

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        X_validation=X_validation,
        y_train=y_train,
        y_test=y_test,
        y_validation=y_validation,
    )


def visualize(data, selected_features, random=False):
    X, y, df, feature_names = load_data(data)
    feature_names.remove("Class")
    np.random.seed(42)  # pour reproductibilité
    if random:
        selected_features = np.random.choice(feature_names, size=3, replace=True)
    print(selected_features)
    print("Features choisies :", selected_features)

    for i in selected_features:
        plt.figure()
        sns.histplot(
            data=df,
            x=i,
            hue=y,  # Sépare les données par la Classe (0 ou 1)
            multiple="stack",  # <--- CHANGEMENT CLÉ : Empile les barres au lieu de les superposer
            bins=30,
            palette="bwr",  # Palette Bleu/Rouge
            edgecolor="black",  # Ajout d'une bordure pour mieux délimiter les barres
        )
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
    #print(preprocessing(data_spam, test_size, validation_size))
    visualize(data_diabetes, [], random=True)
