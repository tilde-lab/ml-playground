"""
Test GAT on CrystalGraphDataset
"""
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from gat_regression_model import GAT
from datasets.crystal_graph_dataset import CrystalGraphDataset
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score

mean_absolute_error = MeanAbsoluteError()
r2 = R2Score()

dataset = CrystalGraphDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_data = torch.utils.data.Subset(dataset, range(train_size))
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=15759, shuffle=False, num_workers=0)

model = GAT(4, 8)
model.load_state_dict(torch.load(r'/root/projects/ml-playground/models/GAT/weights/weights01_01.pth'))

total_loss = 0
num_samples = 0

model.eval()
with torch.no_grad():
    mean_loss = 0
    cnt = 0
    for data, y in test_dataloader:
        cnt += 1
        out = model(data)
        loss = F.mse_loss(out, y)

        mean_absolute_error.update(torch.tensor(out.tolist()).reshape(-1), torch.tensor(y.tolist()))
        mae_result = mean_absolute_error.compute()

        r2.update(torch.tensor(torch.tensor(out.tolist())).reshape(-1), torch.tensor(y.tolist()))
        r2_res = r2.compute()

print("R2: ", r2_res, " MAE: ", mae_result, "Pred from", out.min(), " to ", out.max())
