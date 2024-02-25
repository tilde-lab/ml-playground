import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import scatter
from tqdm import tqdm
from descriptors.seebeck_coefficient.graph_from_structure import MolecularGraphDataset

dataset = MolecularGraphDataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_data, batch_size=1024, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=1024, shuffle=False, num_workers=0)

class GCN(torch.nn.Module):
  """Graph Convolutional Network"""

  def __init__(self):
      super().__init__()
      self.conv1 = GCNConv(121, 16)
      self.conv2 = GCNConv(16, 8)
      self.layer3 = Linear(8, 1)


  def forward(self, data):
      x, edge_index = data.x, data.edge_index.type(torch.int64)

      x = self.conv1(x, edge_index)
      x = F.relu(x)
      x = F.dropout(x, training=self.training)
      x = self.conv2(x, edge_index)
      x = F.relu(x)
      x = scatter(x, data.batch, dim=0, reduce='mean')

      x = self.layer3(x)
      return x


device = torch.device('mps')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(300)):
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
    print(f'--------Mean loss for epoch {epoch} is {mean_loss/cnt}--------')

model.eval()
total_loss = 0
num_samples = 0

with torch.no_grad():
    for data, y in test_dataloader:
        pred = model(data.to(device))
        loss = F.mse_loss(pred, y.to(device))
        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs

mse = total_loss / num_samples
print("Mean Squared Error:", mse)