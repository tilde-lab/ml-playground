"""
Training a TransformerModel to predict the Seebeck value.

Transformer made from encoder (without decoder). Uses token-vector to represent embedding for each crystal.
Tensor of tokens is fed to Sequential. Next, loss is calculated as in standard models.
"""
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score

class TransformerModel(nn.Module):
    """A transformer model. Contains an encoder (without decoder)"""
    def __init__(self, n_feature, heads):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=n_feature, nhead=heads, batch_first=True,
                                                   activation="gelu", dropout=0, norm_first=True)
        # size (1, 1, n_feature)
        self.agg_token = torch.rand((1, 1, n_feature))
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)
        self.layer1 = nn.Linear(n_feature, 4 * n_feature)
        self.layer2 = nn.Linear(4 * n_feature, 8 * n_feature)
        self.layer3 = nn.Linear(8 * n_feature, 1)
        self.activ = nn.ELU()

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
            token_emb = x[:, 0]
            emb_list.append(token_emb)

        x = self.layer1(emb_list[0])
        x = self.activ(x)
        x = self.layer2(x)
        x = self.activ(x)
        x = self.layer3(x)
        return x

if __name__ == '__main__':
    r2 = R2Score()
    mean_absolute_error = MeanAbsoluteError()

    total = pd.read_csv(
        '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/uniq_vectors_str_200.csv'
    )
    seebeck = pd.read_csv(
        '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/seebeck_200.csv'
    )
    dataset = pd.concat([seebeck["Seebeck coefficient"], total], axis=1).values.tolist()

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    train_data = torch.utils.data.Subset(dataset, range(train_size))
    test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
    model = TransformerModel(n_feature=2, heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model.train()
    for epoch in tqdm(range(20)):
        mean_loss = 0
        cnt = 0
        for y, atoms, dist in train_data:
            data = [eval(atoms), eval(dist)]
            if len(data[0]) == 0:
                continue
            cnt += 1
            optimizer.zero_grad()
            out = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
            loss = F.mse_loss(out, torch.tensor(y))
            loss.backward()
            optimizer.step()

            mean_loss += loss
        print(f'--------Mean loss for epoch {epoch} is {mean_loss / cnt}--------')

        if epoch % 1 == 0:
            torch.save(
                model.state_dict(),
                f'/root/projects/ml-playground/models/transformer/weights/weights02_02.pth'
            )

    model.eval()
    total_loss = 0
    num_samples = 0

    preds = []
    y_s = []
    with torch.no_grad():
        mean_loss = 0
        cnt = 0
        for y, atoms, dist in test_data:
            data = [eval(atoms), eval(dist)]
            if len(data[0]) == 0:
                continue
            cnt += 1
            pred = model([torch.tensor(data).permute(1, 0).unsqueeze(0)])
            preds.append(pred)
            y_s.append(y)

    mean_absolute_error.update(torch.tensor(preds).reshape(-1), torch.tensor(y_s))
    mae_result = mean_absolute_error.compute()

    r2.update(torch.tensor(preds).reshape(-1), torch.tensor(y_s))
    r2_res = r2.compute()

    torch.save(
        model.state_dict(),
        f'/root/projects/ml-playground/models/transformer/weights/weights02_02.pth'
    )

    print("R2: ", r2_res, " MAE: ", mae_result)
