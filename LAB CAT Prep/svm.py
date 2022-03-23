from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score

#Load dataset
cancer = datasets.load_breast_cancer()
# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("\nLabels: ", cancer.target_names)

# print data(feature)shape
print(cancer.data.shape)
# print the cancer data features (top 5 records)
print("\n",cancer.data[0:5])
# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy: ",accuracy_score(y_test, y_pred))
print("Other Parameters: \n",classification_report(y_test, y_pred))