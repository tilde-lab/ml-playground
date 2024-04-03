import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter
from tqdm import tqdm
from datasets.vectors_graph_dataset import CrystalGraphVectorsDataset
from torchmetrics import MeanAbsoluteError
from torch_geometric.nn import GATv2Conv
from torcheval.metrics import R2Score


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_ch, hidden=16, hidden2=8, activation='elu'):
        super().__init__()
        self.conv1 = GATv2Conv(in_ch, hidden, 1, edge_dim=1)
        self.conv2 = GATv2Conv(hidden, hidden2, 1, edge_dim=1)
        self.layer3 = Linear(hidden2, 1)
        if activation == 'elu':
            self.activ = F.elu
        elif activation == 'relu':
            self.activ = F.relu
        elif activation == 'leaky_relu':
            self.activ = F.leaky_relu
        elif activation == 'tanh':
            self.activ = F.tanh

    def forward(self, data):
        edge_attr = None
        try:
          x, edge_index, edge_attr = data.x, data.edge_index.type(torch.int64), data.edge_attr
        except:
          x, edge_index = data.x, data.edge_index.type(torch.int64)

        x = self.conv1(
            x.float(), edge_index=edge_index, edge_attr=edge_attr
        )
        x = self.activ(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.activ(x)

        x = scatter(x, data.batch, dim=0, reduce='mean')

        x = self.layer3(x)
        return x

    def fit(self, model, train_dataloader, optimizer, device):
        model.train()
        for epoch in tqdm(range(5)):
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
        print(f'--------Mean loss for epoch {epoch} is {mean_loss / cnt}--------')
        torch.save(
            model.state_dict(),
            f'/root/projects/ml-playground/models/GAT/weights/weights04_02.pth'
        )
    def val(self, model, test_dataloader, device):
        r2 = R2Score()
        mean_absolute_error = MeanAbsoluteError()

        model.eval()
        with torch.no_grad():
            cnt = 0
            for data, y in test_dataloader:
                cnt += 1
                pred = model(data.to(device))
                mean_absolute_error.update(
                    torch.tensor(pred.tolist()).reshape(-1), torch.tensor(y.tolist())
                )
                mae_result = mean_absolute_error.compute()

                r2.update(torch.tensor(torch.tensor(pred.tolist())).reshape(-1),
                          torch.tensor(y.tolist()))
                r2_res = r2.compute()
        print("R2: ", r2_res, " MAE: ", mae_result, "Pred from", pred.min(), " to ", pred.max())

        return r2_res, mae_result


if __name__ == '__main__':
    # dataset with atoms and distance info
    dataset = CrystalGraphVectorsDataset()

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=5240, shuffle=False, num_workers=0)

    device = torch.device('cpu')
    model = GAT(in_ch=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model.fit(model, train_dataloader, optimizer, device)
    model.val(model, test_dataloader, device)

    torch.save(
        model.state_dict(),
        f'/root/projects/ml-playground/models/GAT/weights/weights02_02.pth'
    )

