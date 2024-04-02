import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import scatter
from tqdm import tqdm
from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score


class GCN(torch.nn.Module):
  """Graph Convolutional Network"""

  def __init__(self, n_hid):
      super().__init__()
      self.conv1 = GCNConv(2, n_hid)
      self.conv2 = GCNConv(n_hid, n_hid)
      self.layer3 = Linear(n_hid, 1)
      self.activ = F.elu

  def forward(self, data):
      x, edge_index = data.x, data.edge_index.type(torch.int64)

      x = self.conv1(x.float(), edge_index)
      x = self.activ(x)
      x = F.dropout(x, training=self.training)

      x = self.conv2(x, edge_index)
      x = self.activ(x)

      x = scatter(x, data.batch, dim=0, reduce='mean')

      x = self.layer3(x)
      return x

if __name__ == '__main__':
    r2 = R2Score()
    mean_absolute_error = MeanAbsoluteError()
    dataset = CrystalGraphVectorsDataset()

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=15759, shuffle=False, num_workers=0)

    device = torch.device('cpu')
    model = GCN(n_hid=16).to(device)

    # model.load_state_dict(torch.load(f'/root/projects/ml-playground/models/GCN/weights/weights02_01.pth'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in tqdm(range(30)):
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
        if epoch % 1 == 0:
            torch.save(
                model.state_dict(),
                f'/root/projects/ml-playground/models/GCN/weights/weights03_01.pth'
            )

    model.eval()
    total_loss = 0
    num_samples = 0

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

    torch.save(
        model.state_dict(),
        f'/root/projects/ml-playground/models/GCN/weights/weights03_01.pth'
    )

    print("R2: ", r2_res, " MAE: ", mae_result)

