#Q2 - Multi-Linear Regression
#By: Jonathan Rufus Samuel - 20BCT0332

import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn import preprocessing

boston = load_boston()
# Creating p DataFrames
df = pd.DataFrame(data= boston.data, columns= boston.feature_names)
target_df = pd.DataFrame(data= boston.target, columns= ['prices'])
df = pd.concat([df, target_df], axis= 1)

print(df.head(5))
print(df.shape)
print(df.dtypes)
print(df.info())
print()

X= df.drop(labels= 'prices', axis= 1)
Y= df['prices']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.4, random_state= 60)
lr = LinearRegression()

#Model Training
lr.fit(X_train, Y_train)
print("X values for predicting :\n")
print(X_test)
predict = lr.predict(X_test)
print("\nPredicted y values are: ")
print(predict)
mi_ma_sca = preprocessing.MinMaxScaler()
col_sel = ['CRIM', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'B', 'LSTAT']
X = df.loc[:,col_sel]
Y = df['prices']
X = pd.DataFrame(df, columns=col_sel)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()

for i, j in enumerate(col_sel):
    sea.regplot(color='green',y=Y, x=X[j], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.45, h_pad=5.0)
plt.show()

print("Model Performance: ")
print('Mean Absolute Error:', mean_absolute_error(Y_test, predict))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test, predict)))
print('Mean Squared Error:', mean_squared_error(Y_test, predict))