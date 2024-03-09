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

train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

class GCN(torch.nn.Module):
  """Graph Convolutional Network"""

  def __init__(self):
      super().__init__()
      self.layer0_one_hot_emb = Linear(1, 4)
      self.conv1 = GCNConv(7, 16)
      self.conv2 = GCNConv(16, 8)
      self.layer3 = Linear(8, 1)


  def forward(self, data):
      x, edge_index = data.x, data.edge_index.type(torch.int64)

      atoms_vector = x[:, :1]
      atoms_xyz = x[:, 1:]

      one_hot_emb = self.layer0_one_hot_emb(atoms_vector)

      x = self.conv1(torch.cat((one_hot_emb, atoms_xyz), dim=1), edge_index)
      x = F.tanh(x)

      x = self.conv2(x, edge_index)
      x = F.tanh(x)
      x = scatter(x, data.batch, dim=0, reduce='sum')

      x = self.layer3(x)
      return x

device = torch.device('mps')
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_list = []

model.train()
for epoch in tqdm(range(50)):
    mean_loss = 0
    cnt = 0
    for data, y in train_dataloader:
        cnt += 1
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        mean_loss += loss
    writer.add_scalar('Loss/train', mean_loss/cnt, epoch)
    loss_list.append(mean_loss)
    print(f'--------Mean loss for epoch {epoch} is {mean_loss/cnt}--------')

model.eval()
total_loss = 0
num_samples = 0

with torch.no_grad():
    mean_loss = 0
    cnt = 0
    for data, y in test_dataloader:
        cnt += 1
        pred = model(data)
        loss = F.mse_loss(
            pred,
            y
        )
        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs
        mean_loss += loss

        mean_absolute_error.update(torch.tensor(pred.tolist()).reshape(-1), torch.tensor(y.tolist()))
        mae_result = mean_absolute_error.compute()

    writer.add_scalar('Loss/test', mean_loss/cnt)

mse = total_loss / num_samples
torch.save(
    model.state_dict(),
    f'/root/projects/ml-playground/models/GCN/weights/weights362835_05.pth'
)
writer.close()

print(f"Mean Squared Error: {mse}", f"\nMAE: {mae_result}")

