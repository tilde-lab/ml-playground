import optuna
from gat_regression_model import GAT
import torch
from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset
from torch_geometric.loader import DataLoader


def objective(trial):
    dataset = CrystalGraphVectorsDataset()

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=5240, shuffle=False, num_workers=0)

    device = torch.device('cpu')

    hidden = trial.suggest_categorical('hidden', [8, 16, 32])
    hidden2 = trial.suggest_categorical('hidden2', [8, 16, 32, 64])
    lr = trial.suggest_float('lr', 0.0001, 0.01)
    activ = trial.suggest_categorical('activ', ['leaky_relu', 'relu', 'elu', 'tanh'])

    model = GAT(2, hidden=hidden, hidden2=hidden2, activation=activ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # train and test
    model.fit(model, train_dataloader, optimizer, device)
    r2, mae = model.val(model, test_dataloader, device)

    return r2

study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="maximize")
study.optimize(objective, n_trials=100)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  R2: ", trial.values)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")