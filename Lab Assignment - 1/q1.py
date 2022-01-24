# Q1 - Linear Regression
# By: Jonathan Rufus Samuel - 20BCT0332

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random


df = pd.read_csv("Lab Assignment - 1\weight-height.csv")
X = df['Height']
Y = df['Weight']

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.info())
print()

def estimate_coeff(X_1, Y_1):
    n = np.size(X_1)
    mean_X = np.mean(X_1)
    mean_Y = np.mean(Y_1)
    SS_XY = np.sum(Y_1*X_1) - n*mean_Y*mean_X
    SS_XX = np.sum(X_1*X_1) - n*mean_X*mean_X
    a_1 = SS_XY / SS_XX
    a_0 = mean_Y - a_1*mean_X
    return (a_0, a_1)

def regression_line(X_2, Y_2, a):
    plt.scatter(X_2, Y_2, color="green", marker="o", s=30)
    Y_p = a[0] + a[1]*X_2
    plt.plot(X_2, Y_p, color="black")
    plt.title('Scatter WITH Regression Line', size=14)
    plt.xlabel('Height (inches)', size=14)
    plt.ylabel('Weight (pounds)', size=14)
    plt.show()

def predict(X_3, Y_3, a):
    plt.scatter(X_3, Y_3, color="m", marker="o", s=30)
    Y_p = a[0] + a[1]*X_3
    plt.plot(X_3, Y_p, color="r")
    plt.title('Initial Scatter Plot + Regression Line: Relationship b/w Height and Weight', size=18)
    plt.xlabel('Height (inches)', size=14)
    plt.ylabel('Weight (pounds)', size=14)

    p_Y = []
    print("Predicted Y Values of 5 random X: \n")
    X2 = random.sample(sorted(X), 5)
    for i in X2:
        X1 = i
        print("value of height: ", end='')
        print(X1, end='')
        print(" inches")
        Y1_p = a[0] + a[1]*X1
        p_Y.append(Y1_p)
        print("predicted value of weight are: ",
              "{0:.4f}".format(Y1_p), " lbs\n")

    Y_3 = Y_3.tolist()
    error = []
    for i in range(len(p_Y)):
        diff = p_Y[i] - Y[i]
    e = (diff/Y[i])*100
    error.append(e)
    print("Error: ")
    print(error)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40, random_state=1)
a = estimate_coeff(X_train, Y_train)

plt.scatter(X_train, Y_train, color="b", marker="o", s=30)
Y_pred = a[0] + a[1]*X
plt.title('Initial Scatter Plot with Relationship b/w Height and Weight', size=14)
plt.xlabel('Height (inches)', size=14)
plt.ylabel('Weight (pounds)', size=14)
plt.show()

print("Estimated coefficients:\nc = ", "{0:.4f}".format(a[0]), ",\n"
      "m = ", "{0:.4f}".format(a[1]))

#regression line plot fucntion call
regression_line(X_train, Y_train, a)

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

# plotting predicted value of weight for height x
predict(X_test, Y_test, a)