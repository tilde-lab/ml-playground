"""
Any of the below regression models can be used for predicting Seebeck Coefficient values of binary compounds.
"""
import turicreate as tc
from turicreate import SFrame
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score
import torch
from datasets.crystal_graph_dataset import CrystalGraphDataset
import pandas as pd
from torch_geometric.loader import DataLoader

mean_absolute_error = MeanAbsoluteError()
metric = R2Score()

dataset = CrystalGraphDataset(cut=True)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_data = torch.utils.data.Subset(dataset, range(train_size))
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

tr, val = [], []
cnt = 0

for d, y in train_dataloader:
    tr.append([[], [], [], [], y.tolist()[0]])
    for row in d.x.tolist():
        tr[cnt][0].append(row[0])
        tr[cnt][1].append(row[1])
        tr[cnt][2].append(row[2])
        tr[cnt][3].append(row[3])
    cnt += 1


cnt = 0
for d, y in test_dataloader:
    val.append([[], [], [], [], y.tolist()[0]])
    for row in d.x.tolist():
        val[cnt][0].append(row[0])
        val[cnt][1].append(row[1])
        val[cnt][2].append(row[2])
        val[cnt][3].append(row[3])
    cnt += 1
    
train = pd.DataFrame(tr, columns=["n", "x", "y", "z", "Seebeck coefficient"])
test = pd.DataFrame(val, columns=["n", "x", "y", "z", "Seebeck coefficient"])
features = ["n", "x", "y", "z"]

# LINEAR REGRESSION MODEL
train_r, test_r = SFrame(train), SFrame(test)
model_linear = tc.linear_regression.create(
    train_r, target="Seebeck coefficient", features=features
)
coefficients_linear = model_linear.coefficients
predictions_linear = model_linear.predict(test_r)

metric.update(torch.tensor(predictions_linear), torch.tensor(test_r["Seebeck coefficient"]))
r2_res_r = metric.compute()
mean_absolute_error.update(torch.tensor(predictions_linear), torch.tensor(test_r["Seebeck coefficient"]))
mae_result_r = mean_absolute_error.compute()
model_linear.summary()

# DECISION TREE MODEL
train_d, test_d = SFrame(train), SFrame(test)
model_decision = tc.decision_tree_regression.create(
    train_d, target="Seebeck coefficient", features=features
)
predictions_decision = model_decision.predict(test_d)

featureimp_decision = model_decision.get_feature_importance()
metric.update(torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"]))
r2_res_d = metric.compute()
mean_absolute_error.update(torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"]))
mae_result_d = mean_absolute_error.compute()
model_decision.summary()

# BOOSTED TREES MODEL
train_b, test_b =  SFrame(train), SFrame(test)
model_boosted = tc.boosted_trees_regression.create(
    train_b, target="Seebeck coefficient", features=features
)
predictions_boosted = model_boosted.predict(test_b)
results_boosted = model_boosted.evaluate(test_b)
featureboosted = model_boosted.get_feature_importance()
metric.update(torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"]))
r2_res_b = metric.compute()
mean_absolute_error.update(torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"]))
mae_result_b = mean_absolute_error.compute()
model_boosted.summary()

# RANDOM FOREST MODEL
train_r, test_r = SFrame(train), SFrame(test)
model_random = tc.random_forest_regression.create(
    train_r, target="Seebeck coefficient", features=features
)
predictions_random = model_random.predict(test_r)
results_random = model_random.evaluate(test_r)
featureimp_random = model_random.get_feature_importance()
metric.update(torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"]))
r2_res_rf = metric.compute()
mean_absolute_error.update(torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"]))
mae_result_rf = mean_absolute_error.compute()
model_random.summary()
