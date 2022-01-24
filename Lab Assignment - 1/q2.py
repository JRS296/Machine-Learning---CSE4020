#Q2 - Multi-Linear Regression
#By: Jonathan Rufus Samuel - 20BCT0332

import pandas as pd
import numpy as np
import seaborn as sea
import math
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn import preprocessing

boston = load_boston()
# Creating p DataFrames
boston_df = pd.DataFrame(data= boston.data, columns= boston.feature_names)
target_df = pd.DataFrame(data= boston.target, columns= ['prices'])
boston_df = pd.concat([boston_df, target_df], axis= 1)
# Variables
X= boston_df.drop(labels= 'prices', axis= 1)
Y= boston_df['prices']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.35, random_state= 80)
lr = LinearRegression()
# Training/Fitting the Model
lr.fit(X_train, Y_train)
print("X values for predicating :\n")
print(X_test)
predicate = lr.predict(X_test)
print("\npredicted y values are :\n")
print(predicate)
mi_ma_sca = preprocessing.MinMaxScaler()
col_sel = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'B', 'LSTAT']
X = boston_df.loc[:,col_sel]
Y = boston_df['prices']
X = pd.DataFrame(boston_df, columns=col_sel)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, j in enumerate(col_sel):
 sea.regplot(y=Y, x=X[j], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# Evaluating Model's Performance
print('Mean Absolute Error:', mean_absolute_error(Y_test, predicate))
print('Mean Squared Error:', mean_squared_error(Y_test, predicate))
print('Mean Root Squared Error:', np.sqrt(mean_squared_error(Y_test, predicate)))