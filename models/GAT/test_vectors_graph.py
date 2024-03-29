"""
Train GAT on CrystalGraphVectorsDataset
"""
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset
from gat_regression_model import GAT
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score

r2 = R2Score()
mean_absolute_error = MeanAbsoluteError()

dataset = CrystalGraphVectorsDataset()
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_data = torch.utils.data.Subset(dataset, range(train_size))
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=5000, shuffle=False, num_workers=0)

model = GAT(2)
model.load_state_dict(torch.load(r'/root/projects/ml-playground/models/GAT/weights/weights01_03.pth'))

model.eval()
with torch.no_grad():
    mean_loss = 0
    cnt = 0
    for data, y in test_dataloader:
        cnt += 1
        pred = model(data)
        loss = F.mse_loss(pred, y)

        mean_absolute_error.update(torch.tensor(pred.tolist()).reshape(-1), torch.tensor(y.tolist()))
        mae_result = mean_absolute_error.compute()

        r2.update(torch.tensor(torch.tensor(pred.tolist())).reshape(-1),
                  torch.tensor(y.tolist()))
        r2_res = r2.compute()

print("R2: ", r2_res, " MAE: ", mae_result, "Pred from", pred.min(), " to ", pred.max())
