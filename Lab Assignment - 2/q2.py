# Decision Tree
from sklearn.metrics import accuracy_score,classification_report,multilabel_confusion_matrix
import matplotlib.pyplot as plt
from stringprep import c22_specials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
import pandas as pd
import seaborn as sns
import numpy as np

fruits_df = pd.read_csv("Lab Assignment - 2/fruit_new.csv")
features_names = ['mass', 'width', 'height', 'color_score']
X = fruits_df[features_names]
y = fruits_df['fruit_label']
print(fruits_df.head(10))

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.45,random_state=10)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
val = tree.DecisionTreeClassifier(criterion='gini')
val = val.fit(X_train, y_train)
y_pred = val.predict(X_test)
print("Predictions: ",y_pred)

c = multilabel_confusion_matrix(y_test,y_pred)
i=0
print("Confusion Matrix: ")
for i in range(4):
    x = c[i]
    print(x)
    grp_names = ['True Negatives','False Positives','False Negatives','True Positives']
    grp_counts = ["{0:0.0f}".format(value) for value in x.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2, in zip(grp_names,grp_counts)] 
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(x, annot=labels, fmt='', cmap="Dark2")
    plt.show()

print('\nAccuracy: {} \n'.format(accuracy_score(y_test,y_pred)))
print("Decision Tree CART (Before Pruning): ")
tree.plot_tree(val)
plt.show()

#Pruning
val2 = tree.DecisionTreeClassifier(random_state=10, ccp_alpha= 4.55)
val2 = val.fit(X_train, y_train)
print("Decision Tree CART (After Pruning): ")
tree.plot_tree(val2)
plt.show()
print()

target_labels = ['Picked','Not Picked']
print(classification_report(y_pred, y_test, target_names = features_names))