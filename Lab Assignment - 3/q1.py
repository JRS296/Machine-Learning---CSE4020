import pandas as pd
import numpy as np

df = pd.DataFrame([[-1, -1, -1],[-1, 1, -1],[1, -1, -1],[1, 1, 1]])
print(df.head(),'\n')

x = df.drop(df.columns[-1],axis = 1)
print(x,'\n')

y = pd.DataFrame(df[df.columns[-1]])
print(y,'\n')

n,m = x.shape # n = number of rows and m = number of columns
print("Number of rows: ",n)
print("Number of columns: ",m)

# initial declaration of weights and bias
w = [-1 for i in range (m)]
b = -1
print("Initial Weights: ",w)
print("Initial Bias:", b)

#initial learning rate and treshold value
r =1
theta = 0
print("Learning rate: ",r)
print("Treshold value: ",theta)

#calculation
count = 0
w_new = w.copy()
b_new = b
init = True
sum_xy = 0
while init or ( sum_xy!= 0 and count <100 ):
    w = w_new.copy()
    b = b_new
    for i in range (n):
        zin = 0
        for j in range (m):
            zin += x.iat[i,j]*w[j]
        zin += b

        if zin > theta:
            out = 1
        else:
            out = -1
        for j in range (m):
            if(y.iat[1,0] != out):
                w_new[j] = w[j] + r*(y.iat[i,0] - out)*x.iat[i,j]
                b_new = b + r*y.iat[i,0]
    init = False
    count+=1
    print("Epoch: ",count," New weights: ",w_new," New Bias: ",b_new)
    sum_xy = sum(w_new)
    sum_xy -= sum(w)