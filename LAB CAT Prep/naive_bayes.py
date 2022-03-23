# load the iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data= iris.data, columns= iris.feature_names)
print(df.head(5))
print(df.shape)
print(df.dtypes)
print(df.info())
print()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# training the model on training set

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

print(classification_report(y_predict_NB, test_y, target_names = target_labels))
print("Accuracy for given MLP Classfier Model: ", accuracy_score(y_predict_NB, test_y))