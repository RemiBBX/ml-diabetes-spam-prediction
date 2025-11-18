import torch.nn
from fontTools.misc.classifyTools import classify
from torch.utils.data import DataLoader, TensorDataset

from src.neural_network import MLP_nn, train_model
from src.preprocessing import preprocessing, data_diabetes
from src.utils import compute_accuracy
from sklearn.metrics import confusion_matrix, classification_report

input_size = 21
train_size = 0.7
validation_size = 0.15
test_size = 0.15

learning_rate = 0.001
epochs = 10

hidden_size = 128


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


    pos = torch.count_nonzero(y_train)
    zeros = X_train.shape[0] - pos
    pos_weight = torch.tensor([len(y_train) / pos], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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

    probs = torch.sigmoid(MLP(X_test)).detach().cpu().numpy().flatten()
    y_true = y_test.numpy().flatten()

    #thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
    thresholds = [0.5]

    for t in thresholds:
        preds = (probs >= t).astype(int)
        print(f"Threshold = {t}")
        print(classification_report(y_true, preds))
