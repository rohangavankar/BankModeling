import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from ProjectBank import do_etl
# Importing the dataset
dataset = do_etl("/Users/rohangavankar/PycharmProjects/test_app/bank/bank-full.csv")
X = dataset.loc[:, ["age","marital_no","education_no","balance","poutcome_no","duration"]].values
y = dataset.loc[:, ['y_no']].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting Decision Tree Classification to the Training set
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy is:", round(accuracy_score(y_test, y_pred), 4))
importance = classifier.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

tn, fp, fn, tp = cm[0][0],cm[0][1] , cm[1][0], cm[1][1]
print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")

recall = tp/(tp+fn)
precision =  tp/(tp+fp)
print("Recall is:" , recall)
print("Precesion is:" , precision)
print('F-Measure is:',  2*((recall*precision)/(recall+precision)))

feature_names = ['Age', 'Salary']
tree.plot_tree(classifier, feature_names = feature_names )
plt.show()