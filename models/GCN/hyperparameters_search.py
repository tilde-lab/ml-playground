"""
Selection of hyperparameters for GCN.
"""
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import scatter
from tqdm import tqdm
from datasets.molecular_graph_dataset import CrystalGraphDataset
import numpy as np
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score


class GCN(torch.nn.Module):
  """Graph Convolutional modelwork"""

  def __init__(self, n_hidden, activation):
      super().__init__()
      self.conv1 = GCNConv(4, n_hidden)
      self.conv2 = GCNConv(n_hidden, n_hidden)
      self.layer3 = Linear(n_hidden, 1)
      if activation == 'elu':
          self.activ = F.elu
      elif activation == 'relu':
          self.activ = F.relu
      elif activation == 'leaky_relu':
          self.activ = F.leaky_relu
      elif activation == 'tanh':
          self.activ = F.tanh

  def forward(self, data):
      x, edge_index = data.x, data.edge_index.type(torch.int64)

      x = self.conv1(x, edge_index)
      x = self.activ(x)
      x = F.dropout(x, training=self.training)

      x = self.conv2(x, edge_index)
      x = self.activ(x)

      x = scatter(x, data.batch, dim=0, reduce='mean')

      x = self.layer3(x)
      return x

def train(model, opt, lr, b_size, max_ep, train_data):
    train_dataloader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=0)

    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    elif opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # else error

    for epoch in tqdm(range(0, max_ep)):
        mean_loss = 0
        cnt = 0
        for data, y in train_dataloader:
            cnt += 1
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, y)
            mean_loss += loss.item()
            loss.backward()
            optimizer.step()
            mean_loss += loss

    print("Training done ")
    return model

def test(model, test_dataloader):
    r2 = R2Score()
    mae = MeanAbsoluteError()

    total_loss = 0
    num_samples = 0

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

            mae.update(torch.tensor(pred.tolist()).reshape(-1), torch.tensor(y.tolist()))
            mae_result = mae.compute()

            r2.update(torch.tensor(torch.tensor(pred.tolist())).reshape(-1),
                      torch.tensor(y.tolist()))
            r2_res = r2.compute()

        mse = total_loss / num_samples

        return mae_result, r2_res.tolist(), mse

def create_params(seed=1):
    """
    Create params:
    n_hidden - hidden layers in the specified range
    activation - activation function
    opt - optimizer
    b_size - batch_size
    max_ep - total number of epoch
    """
    rnd = np.random.RandomState(seed)

    n_hidden = rnd.randint(6, 32)
    activation = rnd.choice(['leaky_relu', 'relu', 'elu', 'tanh'])

    opt = 'adam' # Adam is better than other
    lr = rnd.uniform(low=0.001, high=0.10)
    b_size = rnd.randint(64, 1024)
    max_ep = rnd.randint(30, 35)

    return (n_hidden, activation, opt, lr, b_size, max_ep)

def search_params(test_data, train_data):

    max_trials = 25
    result = []

    for i in range(max_trials):
        print("\nSearch trial " + str(i))
        (n_hidden, activation, opt, lr, b_size, max_ep) = \
        create_params(seed=i*24)

        result.append([[n_hidden, activation, opt, lr, b_size, max_ep]])
        print((n_hidden, activation, opt, lr, b_size, max_ep))

        model = GCN(n_hidden, activation)

        model.train()
        model = train(model, opt, lr, b_size, max_ep, train_data)

        model.eval()
        test_dataloader = DataLoader(test_data, batch_size=15759, shuffle=False, num_workers=0)
        mae_result, r2_res, mse = test(model, test_dataloader)
        result[i].append([mae_result, r2_res, mse])
        print(f"MAE: {mae_result}, R2: {r2_res}, MSE: {mse}")

    return result

def main():
    print("\nBegin hyperparameter random search ")

    dataset = CrystalGraphDataset()

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

    result = search_params(train_data, test_data)

    for res in result:
        print(res)

if __name__ == "__main__":
    main()