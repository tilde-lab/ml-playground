import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import scatter
from tqdm import tqdm
from datasets.molecular_graph_dataset import MolecularGraphDataset
import pickle
from tensorboardX import SummaryWriter
import numpy as np
from torchmetrics import MeanAbsoluteError

mean_absolute_error = MeanAbsoluteError()
dataset = MolecularGraphDataset()

writer = SummaryWriter('logs')

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_data = torch.utils.data.Subset(dataset, range(train_size))
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

train_dataloader = DataLoader(train_data, batch_size=524, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=15759, shuffle=False, num_workers=0)

class GCN(torch.nn.Module):
  """Graph Convolutional Network"""

  def __init__(self):
      super().__init__()
      self.conv1 = GCNConv(4, 16)
      self.conv2 = GCNConv(16, 8)
      self.layer3 = Linear(4, 8)
      self.layer4 = Linear(16, 1)

  def forward(self, d):
      # k, p, c, b - additional properties for total graph (see ml-playground/data_massage/props.json)
      data, y, p, c, apf, wiener = d
      p = p.view(-1, 1)
      c = c.view(-1, 1)
      apf = apf.view(-1, 1)
      wiener = wiener.view(-1, 1)


      additional_data = torch.cat((p, c, apf, wiener), dim=1)
      x, edge_index = data.x, data.edge_index.type(torch.int64)

      # first input
      x = self.conv1(x[:, :4], edge_index)
      x = F.tanh(x)
      x = F.dropout(x, training=self.training)
      x = self.conv2(x, edge_index)
      x = F.tanh(x)

      # second input
      add_x = self.layer3(additional_data)
      add_x = F.tanh(add_x)

      x = scatter(x, data.batch, dim=0, reduce='sum')

      x = self.layer4(torch.cat((x, add_x), dim=1))
      return x

device = torch.device('cpu')
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_list = []

model.train()
for epoch in tqdm(range(150)):
    mean_loss = 0
    cnt = 0
    for data, y, p, c, apf, wiener in train_dataloader:
        d = [data, y, p, c, apf, wiener]
        cnt += 1
        optimizer.zero_grad()
        out = model(d)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        mean_loss += loss
    writer.add_scalar('Loss/train', mean_loss/cnt, epoch)
    loss_list.append(mean_loss)
    print(f'--------Mean loss for epoch {epoch} is {mean_loss/cnt}--------')
    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            f'/root/projects/ml-playground/models/GCN/weights/weights000005_01.pth'
        )

model.eval()
total_loss = 0
num_samples = 0

with torch.no_grad():
    mean_loss = 0
    cnt = 0
    for data, y, p, c, apf, wiener in test_dataloader:
        d = [data, y, p, c, apf, wiener]
        cnt += 1
        pred = model(d)
        loss = F.mse_loss(pred, y)

        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs
        mean_loss += loss

        mean_absolute_error.update(torch.tensor(pred.tolist()).reshape(-1), torch.tensor(y.tolist()))
        mae_result = mean_absolute_error.compute()

    writer.add_scalar('Loss/test', mean_loss/cnt)

mse = total_loss / num_samples
torch.save(
    model.state_dict(),
    f'/root/projects/ml-playground/models/GCN/weights/weights000005_01.pth'
)
writer.close()

print(f"Mean Squared Error: {mse}", f"\nMAE: {mae_result}")

