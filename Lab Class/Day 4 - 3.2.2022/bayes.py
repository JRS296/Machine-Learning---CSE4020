import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

iris_data = load_iris()

data = iris_data.data
targets = iris_data.target
distances = dict()
# print(data)


def train(data, target):


global distances
counters = dict()
n = len(data)
for i in range(n):
dist = np.linalg.norm(data[i])
x = distances.get(target[i], None)
if x:
counters[target[i]] = counters[target[i]] + 1
distances.update({target[i]: (dist + x)})
else:
distances.update({target[i]: dist})
counters.update({target[i]: 1})
for i in counters.items():
distances[i[0]] = distances[i[0]] / i[1]


def predict(input):


global distances
predictions = np.ndarray(len(input), dtype=int)
c = 0
for j in input:
dist = np.linalg.norm(j)
min_dist = 999
prediction = 0
for i in distances.items():
if (abs(dist - i[1])) < min_dist:
min_dist = abs(dist - i[1])
prediction = i[0]
predictions[c] = prediction
c = c + 1
return predictions

train_data, test_data, train_target, test_target = train_test_split(
    data, targets, test_size=0.33, random_state=101)

train(train_data, train_target)
pred = predict(test_data)
print("Testing data:", test_target)
print("Predictions by knn:", pred)

print('Mean Absolute Error:', mean_absolute_error(test_target, pred))
print('Mean Squared Error:', mean_squared_error(test_target, pred))
print('Mean Root Squared Error:', np.sqrt(
    mean_squared_error(test_target, pred)))
