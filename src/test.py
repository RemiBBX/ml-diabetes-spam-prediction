import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.neural_network import MLP_nn, train_model

# -------------------
# 1. Load dataset
# -------------------
data = load_breast_cancer()
X = torch.tensor(data.data, dtype=torch.float32)
y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)

input_size = X.shape[1]  # 30 pour le dataset breast cancer
hidden_size = 64  # ou ce que tu veux
layers = 3  # nombre de couches cachées
drop_rate = 0.1

MLP = MLP_nn(input_size=input_size, hidden_size=hidden_size, output_size=1, layer_size=layers, drop_rate=drop_rate)
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# -------------------
# 2. Model, criterion, optimizer
# -------------------
# Assumes `MLP` is already defined and imported
device = "cpu"
MLP.to(device)

# Optionnel : gérer léger déséquilibre
pos = y_train.sum()
neg = len(y_train) - pos
pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(MLP.parameters(), lr=0.001)

# -------------------
# 3. Train
# -------------------
train_model(
    model=MLP,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=20,
    print_every_epochs=1,
)

# -------------------
# 4. Evaluate
# -------------------
MLP.eval()
with torch.no_grad():
    logits = MLP(X_test.to(device))
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    y_true = y_test.numpy().flatten().astype(int)

print("Accuracy (test):", accuracy_score(y_true, preds))
print(classification_report(y_true, preds))
