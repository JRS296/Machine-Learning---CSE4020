from re import X
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Lab Assignment - 1\weight-height.csv')
print(df.head())

print(df.shape)
print(df.dtypes)
print(df.info())
print()

df.Gender.nunique()
df.Gender.unique()

ax1 = df[df['Gender'] == 'Male'].plot(kind='scatter', x='Height', y='Weight', color='black', alpha=0.5, figsize=(10, 7))
df[df['Gender'] == 'Female'].plot(kind='scatter', x='Height', y='Weight', color='green', alpha=0.5, figsize=(10 ,7), ax=ax1)
plt.legend(labels=['Males', 'Females'])
plt.title('Initial Scatter Plot: Relationship b/w Height and Weight', size=18)
plt.xlabel('Height (inches)', size=18)
plt.ylabel('Weight (pounds)', size=18)
print("Initial Scatter Plot: ")
plt.show()

df_males = df[df['Gender'] == 'Male']
df_females = df[df['Gender'] == 'Female']
male_fit = np.polyfit(df_males.Height, df_males.Weight, 1)
print("Male Fit Plot: ",male_fit)
female_fit = np.polyfit(df_females.Height, df_females.Weight, 1)
print("Female Fit plot: ",female_fit)
df_males = df[df['Gender'] == 'Male']
df_females = df[df['Gender'] == 'Female']

ax1 = df_males.plot(kind='scatter', x='Height', y='Weight', color='purple', alpha=0.5, figsize=(10, 7))
df_females.plot(kind='scatter', x='Height', y='Weight', color='red', alpha=0.5, figsize=(10, 7), ax=ax1)
plt.plot(df_males.Height, male_fit[0] * df_males.Height + male_fit[1], color='green', linewidth=2)
plt.plot(df_females.Height, female_fit[0] * df_females.Height + female_fit[1], color='gold', linewidth=2)
plt.text(65, 230, 'y={:.2f}+{:.2f}*x'.format(male_fit[1], male_fit[0]), color='darkblue', size=12)
plt.legend(labels=['Males Regresion Line', 'Females Regresion Line', 'Males', 'Females'])
plt.title('Initial Scatter Plot + Regression Line: Relationship b/w Height and Weight', size=18)
plt.xlabel('Height (inches)', size=18)
plt.ylabel('Weight (pounds)', size=18)
print("Initial Scatter Plot + Regression Line: ")
plt.show()

print("Prediction of Y value given X (via Regression)")
X = df.iloc[:, :-1].values 
y = df.iloc[:, 1].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(50)
print("Predicted Result: ",y_pred)