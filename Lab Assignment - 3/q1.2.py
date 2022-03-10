#Q1.2 – Percpetron of OR gate
#By: Jonathan Rufus Samuel - 20BCT0332

# Perceptron Model of AND and OR gate

import pandas as pd
import numpy as np

#Or Gate
print("Perceptron Network for OR Gate:")
df = pd.DataFrame([[1,1,1],[-1, 1, 1],[1, -1, 1],[-1, -1, -1]])
print(df.head(),'\n')

x = df.drop(df.columns[-1],axis = 1)
print(x,'\n')
y = pd.DataFrame(df[df.columns[-1]])
print(y,'\n')

r,c = x.shape 
print("Number of rows: ",r)
print("Number of columns: ",c)
w = [-1 for i in range (c)]
b = -1
print("Initial Weights: ",w)
print("Initial Bias:", b)

rt =1
theta = 0
print("Learning rate: ",rt)
print("Treshold value: ",theta)
count = 0
w_new = w.copy()
b_new = b
init = True
sum_xy = 0
while init or ( sum_xy!= 0 and count <100 ):
    w = w_new.copy()
    b = b_new
    for i in range (r):
        zin = 0
        for j in range (c):
            zin += x.iat[i,j]*w[j]
        zin += b

        if zin > theta:
            out = 1
        else:
            out = -1
        for j in range (c):
            if(y.iat[1,0] != out):
                w_new[j] = w[j] + rt*(y.iat[i,0] - out)*x.iat[i,j]
                b_new = b + rt*y.iat[i,0]
    init = False
    count+=1
    print("Epoch: ",count," New weights: ",w_new," New Bias: ",b_new)
    sum_xy = sum(w_new)
    sum_xy -= sum(w)
