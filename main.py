import torch.nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

from src.neural_network import MLP_nn, train_model
from src.preprocessing import preprocessing

data_diabetes = "./data/diabetes_binary_health_indicators_BRFSS2015.csv"
input_size = 21
train_size = 0.8
validation_size = 0.15
test_size = 0.2

learning_rate = 0.1
epochs = 10

hidden_size = 10


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


    # Changer l'init
    # Verifier la stratify
    # Dropout
    # AdamW
    # cross val


    pos = torch.count_nonzero(y_train)
    zeros = X_train.shape[0] - pos
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=zeros/pos)
    optimizer = torch.optim.AdamW(MLP.parameters(), lr=learning_rate)

    train_model(
        model=MLP,
        train_loader=train_loader,
        val_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        print_every_epochs=1
    )

    y_predict = MLP(X_test)
    y_predict = (y_predict > 0.5).int()
    accu = classification_report(
        y_true=y_test.detach().numpy(),
        y_pred=y_predict.detach().numpy(),
    )

    print(accu)


