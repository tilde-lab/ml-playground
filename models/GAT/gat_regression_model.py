import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter
from tqdm import tqdm
from datasets.molecular_graph_dataset import MolecularGraphDataset
import pickle
from tensorboardX import SummaryWriter
import numpy as np
from torchmetrics import MeanAbsoluteError
from torch_geometric.nn import GATv2Conv


mean_absolute_error = MeanAbsoluteError()
dataset = MolecularGraphDataset()

writer = SummaryWriter('logs')

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_data = torch.utils.data.Subset(dataset, range(train_size))
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, heads=8):
    super().__init__()
    self.layer0_one_hot_emb = Linear(1, 4)
    self.conv1 = GATv2Conv(7, 32, heads=heads, edge_dim=1)
    self.conv2 = GATv2Conv(32*heads, 16, heads=heads, edge_dim=1)
    self.conv3 = GATv2Conv(16*heads, 8, heads=1, edge_dim=1)

    self.layer3 = Linear(8, 1)

  def forward(self,data):
    x, edge_index, edge_attr = data.x, data.edge_index.type(torch.int64), data.edge_attr

    atoms_vector = x[:, :1]
    atoms_xyz = x[:, 1:]

    one_hot_emb = self.layer0_one_hot_emb(atoms_vector)

    x = self.conv1(
        torch.cat((one_hot_emb, atoms_xyz), dim=1), edge_index=edge_index,
        edge_attr=edge_attr
    )
    x = F.tanh(x)
    x = F.dropout(x, p=0.6, training=self.training)

    x = self.conv2(x, edge_index, edge_attr)
    x = F.tanh(x)

    x = self.conv3(x, edge_index, edge_attr)
    x = F.tanh(x)

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

model.eval()
total_loss = 0
num_samples = 0

# load scaler for the reverse normalization operation
with open('/root/projects/ml-playground/data_massage/normalization/scalers/scaler190027.pkl', 'rb') as f:
    scaler = pickle.load(f)

with torch.no_grad():
    mean_loss = 0
    cnt = 0
    for data, y in test_dataloader:
        cnt += 1
        original_data_y = scaler.inverse_transform((y.to(device)).reshape(-1, 1))
        pred = model(data.to(device))
        original_pred = scaler.inverse_transform(pred.cpu().numpy())
        loss = F.mse_loss(
            torch.from_numpy(original_pred),
            torch.from_numpy(original_data_y)
        )
        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs
        mean_loss += loss

        mean_absolute_error.update(torch.tensor(original_pred.tolist()), torch.tensor(original_data_y.tolist()))
        mae_result = mean_absolute_error.compute()

        predictions = np.array(scaler.inverse_transform(pred.cpu().numpy()))
        targets = np.array(original_data_y)

        np.savetxt('predictions.csv', predictions, delimiter=',')
        np.savetxt('targets.csv', targets, delimiter=',')

    writer.add_scalar('Loss/test', mean_loss/cnt)


mse = total_loss / num_samples
torch.save(
    model.state_dict(),
    f'/root/projects/ml-playground/models/GAT/weights/weights190027_02.pth'
)
writer.close()

print(f"Mean Squared Error: {mse}", f"\nMAE: {mae_result}")

