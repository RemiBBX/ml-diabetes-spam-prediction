from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_diabetes = "./data/diabetes_binary_health_indicators_BRFSS2015.csv"

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

def preprocessing(data, test_size, validation_size) :
    df = pd.read_csv(data)
    df = df.fillna(df.median())                     # gérer les valeurs manquantes
    df = df.sample(frac=0.01, random_state=42)      # réduit le nombre de ligne pour des tests


    y = df['Diabetes_binary']
    X = df.drop(columns=['Diabetes_binary'])

    ## Standardisation avec pd
    # df = (df - df.mean()) / df.std()
    # df = (df - df.min()) / (df.max() - df.min())
    ## Standardisation avec scikit
    X = StandardScaler().fit_transform(X)           # moyenne nulle et variance unité
    y = y.to_numpy().reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size = test_size + validation_size,
        random_state=42,
        stratify=y            # Classes en même proportion que dans le dataset entier
    )

    X_test, X_validation, y_test, y_validation = train_test_split(
        X_temp, y_temp,
        test_size=validation_size/(test_size + validation_size),
        random_state=42,
        stratify=y_temp  # classes en même proportion que dans le dataset entier
    )

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        X_validation=X_validation,
        y_train=y_train,
        y_test=y_test,
        y_validation=y_validation,
    )

if __name__ == "__main__":
    print(preprocessing(
        data=data_diabetes,
        test_size=test_size,
        validation_size=validation_size,
    ))
