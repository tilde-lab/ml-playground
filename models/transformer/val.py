import torch
import torch.utils.data as data
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score
import pandas as pd
from transformer_reg import TransformerModel

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

model.load_state_dict(
    torch.load(f'/root/projects/ml-playground/models/transformer/weights/weights02_01.pth')
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

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

print("R2: ", r2_res, " MAE: ", mae_result)
