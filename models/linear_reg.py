from sklearn import linear_model
import numpy as np
from torchmetrics import MeanAbsoluteError
from torcheval.metrics import R2Score
import torch
import pandas as pd

mean_absolute_error = MeanAbsoluteError()
r2 = R2Score()

# Crystal in vectors format
total = pd.read_csv(
    '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/rep_vectors_str_200.csv'
)
seebeck = pd.read_csv(
    '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/seebeck_200.csv'
)
total_transformed = []
seebeck_transformed = [i[0] for i in seebeck.values.tolist()]

for i, row in enumerate(total.values.tolist()):
    atoms = eval(row[0])
    distance = eval(row[1])
    total_transformed.append([l for l in atoms])
    [total_transformed[i].append(k) for k in distance]

train_size = int(0.9 * len(total))
test_size = len(total) - train_size

train_y = np.array(seebeck_transformed[:train_size])
test_y = np.array(seebeck_transformed[test_size:])

train_x = np.array(total_transformed[:train_size])
test_x = np.array(total_transformed[test_size:])

# Create linear regression
regr = linear_model.LinearRegression()

regr.fit(train_x, train_y)

# Make predictions using the testing set
pred = regr.predict(test_x)

r2.update(torch.tensor(pred), torch.tensor(test_y))
r2_res = r2.compute()
mean_absolute_error.update(torch.tensor(pred), torch.tensor(test_y))
mae = mean_absolute_error.compute()

print(f'MAE: {mae}, R2: {r2_res}')
