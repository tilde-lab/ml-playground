"""
Training a TransformerModel to predict the Seebeck value.

Сrystal is represented as a sequence of atoms and is fed to the input of Transformer. Transformer made from encoder
(without decoder). Uses token-vector to represent embedding for each crystal. Tensor of tokens is fed to Sequential.
Next, loss is calculated as in standard models.
"""
import torch
import torch.nn as nn
import torch.utils.data as data
from torch_geometric.loader import DataLoader
from datasets.crystal_graph_dataset import CrystalGraphDataset
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score

r2 = R2Score()
mean_absolute_error = MeanAbsoluteError()

dataset = CrystalGraphDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_data = torch.utils.data.Subset(dataset, range(train_size))
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
train_dataloader = DataLoader(train_data, batch_size=524, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=15759, shuffle=False, num_workers=0)

class TransformerModel(nn.Module):
    """A transformer model. Contains an encoder (without decoder)"""
    def __init__(self, n_feature, heads):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_feature, nhead=heads)
        # size (1, 1, n_feature)
        self.agg_token = torch.rand((1, 1, n_feature))
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.ff = nn.Sequential(
            nn.Linear(n_feature, 4 * n_feature),
            nn.ELU(),
            nn.Linear(4 * n_feature, 8 * n_feature),
            nn.ELU(),
            nn.Linear(8 * n_feature, 1)
        )

    def forward(self, batch):
        """
        'agg_token' is concatenated to every matrix of crystal. Feeding into the transformer
        occurs separately for each crystal. Before transfer to Sequential (self.ff), token embeddings
        are extracted.
        """
        emb_list = []

        for data in batch:
            x = torch.cat([self.agg_token, data], dim=1)
            x = self.transformer_encoder(x)

            # get token embedding
            token_emb = x[:,0]
            emb_list.append(token_emb.tolist())

        x = torch.tensor(emb_list)
        x = self.ff(x)

        return x


def graph_to_seq(data):
    ready_atoms = 0
    x_crystal = data.x.tolist()
    crystals = []

    # in range num of crystal
    for n_crystal in set(data.batch.tolist()):
        num_atoms = data.batch.tolist().count(n_crystal)
        crystals.append([])
        for atom in range(num_atoms):
            crystals[n_crystal].append([])
            for el in range(4):
                crystals[n_crystal][atom].append(x_crystal[ready_atoms][el])
            ready_atoms += 1

    res = [torch.tensor(i).unsqueeze(0) for i in crystals]

    return res

model = TransformerModel(n_feature=4, heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(150)):
    mean_loss = 0
    cnt = 0
    for data, y in train_dataloader:
        cnt += 1
        optimizer.zero_grad()
        data = graph_to_seq(data)

        out = model(data)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()

        mean_loss += loss
    print(f'--------Mean loss for epoch {epoch} is {mean_loss / cnt}--------')

    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            f'/root/projects/ml-playground/models/transformer/weights/weights000007_02.pth'
        )

model.eval()
total_loss = 0
num_samples = 0

with torch.no_grad():
    mean_loss = 0
    cnt = 0
    for data_orig, y in test_dataloader:
        cnt += 1
        data = graph_to_seq(data_orig)
        pred = model(data)
        loss = F.mse_loss(pred, y)

        total_loss += loss.item() * data_orig.num_graphs
        num_samples += data_orig.num_graphs
        mean_loss += loss

        mean_absolute_error.update(torch.tensor(pred.tolist()).reshape(-1), torch.tensor(y.tolist()))
        mae_result = mean_absolute_error.compute()

        r2.update(torch.tensor(torch.tensor(pred.tolist())).reshape(-1),
                  torch.tensor(y.tolist()))
        r2_res = r2.compute()

mse = total_loss / num_samples
torch.save(
    model.state_dict(),
    f'/root/projects/ml-playground/models/transformer/weights/weights000007_02.pth'
)

print("R2: ", r2_res, " MAE: ", mae_result)
