import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from gcn_regression_model import GCN
from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score

r2 = R2Score()
mean_absolute_error = MeanAbsoluteError()

dataset = CrystalGraphVectorsDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
test_dataloader = DataLoader(test_data, batch_size=4000, shuffle=False, num_workers=0)

device = torch.device('cpu')
model = GCN(n_hid=8).to(device)
model.load_state_dict(torch.load(f'/root/projects/ml-playground/models/GCN/weights/weights12_04.pth'))

loss_list = []

model.eval()
with torch.no_grad():
    mean_loss = 0
    cnt = 0
    for data, y in test_dataloader:
        cnt += 1
        pred = model(data.to(device))
        loss = F.mse_loss(pred, y.to(device))

        mean_absolute_error.update(torch.tensor(pred.tolist()).reshape(-1), torch.tensor(y.tolist()))
        mae_result = mean_absolute_error.compute()

        r2.update(torch.tensor(torch.tensor(pred.tolist())).reshape(-1),
                  torch.tensor(y.tolist()))
        r2_res = r2.compute()

print("R2: ", r2_res, " MAE: ", mae_result)

