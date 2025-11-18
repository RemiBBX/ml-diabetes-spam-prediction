from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Layer:
    pass


class MLP_nn(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, layer_size = 3, drop_rate = 0.4):
        super(MLP_nn, self).__init__()
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
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
    scheduler = None

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
        #scheduler.step()

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
            print('epoch: {} \ttraining Loss: {:.6f} '.format(i + 1, train_loss, valid_loss))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_state = model.state_dict()
            print(f"Validation loss improved → new best = {best_val_loss:.6f}")
        else:
            print(
                f"No improvement in validation loss (current: {valid_loss:.6f}, best: {best_val_loss:.6f}) → reverting weights")
            model.load_state_dict(best_state)

    return train_losses, valid_losses


if __name__ == "__main__":
    learning_rate = 0.001
    MLP = MLP_nn(input_size=21, hidden_size=100, output_size=1, layer_size=3)

