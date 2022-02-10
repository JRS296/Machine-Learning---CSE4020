# Importing library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def acuracy_score(predictions, test):
	correct = 0
	for i in range(len(test)):
		if test[i][-1] == predictions[i]:
			correct += 1
	return (correct / float(len(test))) * 100.0

df = pd.read_csv('Lab Assignment - 2\emails.csv')
print(df.head(5)) #Check if CSV file was loaded successfully

print("Ratio between Legitimate/Spam:\n")
count_Class = pd.value_counts(df['Prediction'], sort=True)
count_Class.plot(kind = 'pie',labels=['Legitimate','Spam'], autopct='%1.0f%%,')
plt.title('Legitimate vs Spam')
plt.ylabel('')
plt.show() #Show ratio between Legitimate and Spam Emails

X = df.iloc[:,1:-1]
Y = df.iloc[:,-1].values
print("Shape of X - ", X.shape)
print("Shape of Y - ", Y.shape)
print()
train_x, test_x, train_y, test_y = train_test_split (X,Y, test_size=0.35, random_state=45)
print("Test Size taken = 0.35")
print("Random State taken = 45\n")

m_nb =  MultinomialNB(alpha = 1.9)
m_nb.fit(train_x, train_y)
y_predict_NB = m_nb.predict(test_x)
target_labels = ['Legitimate','Spam']
print(classification_report(y_predict_NB, test_y, target_names = target_labels))
print("Accuracy for given Naive Bayes Classifier Model: ", accuracy_score(y_predict_NB, test_y))

cm = confusion_matrix(test_y, y_predict_NB)
grp_names = ['True Negatives','False Positives','False Negatives','True Positives']
grp_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

labels = [f"{v1}\n{v2}" for v1, v2, in zip(grp_names,grp_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm, annot=labels, fmt='', cmap="BuPu")
plt.show()
print("Confusion Matrix Values: ",cm)

