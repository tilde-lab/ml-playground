import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter
from tqdm import tqdm
from datasets.crystal_graph_dataset import CrystalGraphDataset
from tensorboardX import SummaryWriter
from torchmetrics import MeanAbsoluteError
from torch_geometric.nn import GATv2Conv
from torcheval.metrics import R2Score

r2 = R2Score()
mean_absolute_error = MeanAbsoluteError()

dataset = CrystalGraphDataset()

writer = SummaryWriter('logs')

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_data = torch.utils.data.Subset(dataset, range(train_size))
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=20759, shuffle=False, num_workers=0)

class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, heads=8):
      super().__init__()
      self.conv1 = GATv2Conv(4, 16, heads=heads, edge_dim=1)
      self.conv2 = GATv2Conv(16 * heads, 8, heads=1, edge_dim=1)
      self.layer3 = Linear(8, 1)

  def forward(self,data):
    x, edge_index, edge_attr = data.x, data.edge_index.type(torch.int64), data.edge_attr

    x = self.conv1(
        x, edge_index=edge_index, edge_attr=edge_attr
    )
    x = F.elu(x)
    x = F.dropout(x, p=0.6, training=self.training)

    x = self.conv2(x, edge_index, edge_attr)
    x = F.elu(x)

    x = scatter(x, data.batch, dim=0, reduce='mean')

    x = self.layer3(x)
    return x

device = torch.device('cpu')
model = GAT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(200)):
    mean_loss = 0
    cnt = 0
    for data, y in train_dataloader:
        cnt += 1
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = F.mse_loss(out, y.to(device))
        loss.backward()
        optimizer.step()
        mean_loss += loss
    writer.add_scalar('Loss/train', mean_loss/cnt, epoch)
    print(f'--------Mean loss for epoch {epoch} is {mean_loss/cnt}--------')
    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            f'/root/projects/ml-playground/models/GAT/weights/weights000007_01.pth'
        )

model.eval()
total_loss = 0
num_samples = 0

torch.save(
    model.state_dict(),
    f'/root/projects/ml-playground/models/GAT/weights/weights000007_01.pth'
)

with torch.no_grad():
    mean_loss = 0
    cnt = 0
    for data, y in test_dataloader:
        cnt += 1
        pred = model(data)
        loss = F.mse_loss(pred, y)

        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs
        mean_loss += loss

        mean_absolute_error.update(torch.tensor(pred.tolist()).reshape(-1), torch.tensor(y.tolist()))
        mae_result = mean_absolute_error.compute()

        r2.update(torch.tensor(torch.tensor(pred.tolist())).reshape(-1),
                  torch.tensor(y.tolist()))
        r2_res = r2.compute()

    writer.add_scalar('Loss/test', mean_loss/cnt)


mse = total_loss / num_samples
torch.save(
    model.state_dict(),
    f'/root/projects/ml-playground/models/GAT/weights/weights000007_01.pth'
)
writer.close()

print("R2: ", r2_res, " MAE: ", mae_result)

