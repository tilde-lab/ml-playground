"""
Any of the below regression models can be used for predicting Seebeck Coefficient values of binary compounds.
"""
import turicreate as tc
from turicreate import SFrame
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score
import torch
import pandas as pd

mean_absolute_error = MeanAbsoluteError()
metric = R2Score()

# Crystal in vectors format
total = pd.read_csv(
    '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/rep_vectors_str_200.csv'
)
seebeck = pd.read_csv(
    '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/seebeck_200.csv'
)
total = pd.concat([seebeck["Seebeck coefficient"], total], axis=1)
features = ["atom", "distance"]

train_size = int(0.9 * len(total))
test_size = len(total) - train_size

train = total.iloc[:train_size]
test = total.iloc[test_size:]

# LINEAR REGRESSION MODEL
train_r = SFrame(train)
test_r = SFrame(test)
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
