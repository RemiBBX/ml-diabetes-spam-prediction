import torch.nn
from torch.utils.data import DataLoader, TensorDataset

from src.neural_network import MLP_nn, train_model
from src.preprocessing import preprocessing
from src.utils import compute_accuracy

data_diabetes = "./data/diabetes_binary_health_indicators_BRFSS2015.csv"
input_size = 21
train_size = 0.7
validation_size = 0.15
test_size = 0.15

learning_rate = 0.001
epochs = 15

hidden_size = 50


if __name__ == "__main__":
    MLP = MLP_nn(input_size=input_size, hidden_size=hidden_size, output_size=1, layer_size=3)

    # Optimizer: new_parameters = old_parameters - lr*gradient, with lr the learning rate

    dataset = preprocessing(
        data=data_diabetes,
        test_size=test_size,
        validation_size=validation_size,
    )
    X_train = torch.tensor(dataset.X_train, dtype=torch.float32)
    y_train = torch.tensor(dataset.y_train, dtype=torch.float32)
    X_test = torch.tensor(dataset.X_test, dtype=torch.float32)
    y_test = torch.tensor(dataset.y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=10_000,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=10_000,
        shuffle=True,
    )

    counts = torch.count_nonzero(y_train)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum()
    criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
    optimizer = torch.optim.Adam(MLP.parameters(), lr=learning_rate)

    train_model(
        model=MLP,
        train_loader=train_loader,
        val_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        print_every_epochs=1
    )

    y_predict = MLP.forward(X_test)
    accu = compute_accuracy(
        y_test_list=y_test,
        y_pred_list=y_predict,
    )

    print(accu)


