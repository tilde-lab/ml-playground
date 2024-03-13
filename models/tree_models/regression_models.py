import turicreate as tc
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score
import torch
from data_massage.seebeck_coefficient_and_structure.data_prepearing_for_decision_tree import data_prepearing_from_file

mean_absolute_error = MeanAbsoluteError()
metric = R2Score()

total = data_prepearing_from_file(
    "/data_massage/seebeck_coefficient_and_structure/data/I_C_PEER_INITIO.xlsx"
)

# Any of the below regression models can be used for predicting Seebeck Coefficient values of binary compounds.

# LINEAR REGRESSION MODEL
# if you don't have "I" ans "C" (see ml-playground/data_massage/props.json), remove it
model_linear = tc.linear_regression.create(
    total, target="Seebeck coefficient", features=["APF", "Wiener", "I", "C"]
)
coefficients_linear = model_linear.coefficients
predictions_linear = model_linear.predict(total)
results_linear = model_linear.evaluate(total)
model_linear.summary()

# DECISION TREE MODEL
train_d, test_d = total.random_split(0.9)
model_decision = tc.decision_tree_regression.create(
    train_d, target="Seebeck coefficient", features=["APF", "Wiener", "I", "C"]
)
predictions_decision = model_decision.predict(test_d)
results_decision = model_decision.evaluate(test_d)
featureimp_decision = model_decision.get_feature_importance()
metric.update(torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"]))
mean_absolute_error.update(torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"]))
mae_result1 = mean_absolute_error.compute()
model_decision.summary()

# BOOSTED TREES MODEL
train_b, test_b = total.random_split(0.9)
model_boosted = tc.boosted_trees_regression.create(
    train_b, target="Seebeck coefficient", features=["APF", "Wiener", "I", "C"]
)
predictions_boosted = model_boosted.predict(test_b)
results_boosted = model_boosted.evaluate(test_b)
featureboosted = model_boosted.get_feature_importance()
metric.update(torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"]))
mean_absolute_error.update(torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"]))
mae_result2 = mean_absolute_error.compute()
model_boosted.summary()

metric = R2Score()
# RANDOM FOREST MODEL
train_r, test_r = total.random_split(0.9)
model_random = tc.random_forest_regression.create(
    train_r, target="Seebeck coefficient", features=["APF", "Wiener", "I", "C"]
)
predictions_random = model_random.predict(test_r)
results_random = model_random.evaluate(test_r)
featureimp_random = model_random.get_feature_importance()
metric.update(torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"]))
mean_absolute_error.update(torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"]))
mae_result3 = mean_absolute_error.compute()
model_random.summary()
