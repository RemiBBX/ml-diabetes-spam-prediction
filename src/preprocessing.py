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




    binary_cols = [col for col in df.columns if set(df[col].unique()) <= {0, 1}]
    continuous_cols = [col for col in df.columns if col not in binary_cols]
    scaler = StandardScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
    y = df['Diabetes_binary']
    X = df.drop(columns=['Diabetes_binary']).astype(float)
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
        X_train=X_train.to_numpy(),
        X_test=X_test.to_numpy(),
        X_validation=X_validation.to_numpy(),
        y_train=y_train,
        y_test=y_test,
        y_validation=y_validation,
    )

if __name__ == "__main__":
    dataset = preprocessing(
        data=data_diabetes,
        test_size=test_size,
        validation_size=validation_size,
    )
    print(len(dataset.X_test[0]))
