import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('Lab Assignment - 3\car_evaluation.csv')
print(df.head(5)) #Check if CSV file was loaded successfully
y = df['buying']
X = df.drop(columns = ['buying'])
X, y = make_classification(n_samples=1728, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)


#car_evaluation = pd.read_csv("Lab Assignment - 3\car_evaluation.csv")
#X = car_evaluation.drop(columns = ['decision'])
#y = car_evaluation['decision']
#X, y = make_classification(n_samples=1728, random_state=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
#clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
#clf.predict(X_test)
#clf.score(X_test, y_test)