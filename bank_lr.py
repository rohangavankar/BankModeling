import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from ProjectBank import do_etl
# Importing the dataset
dataset = do_etl("/Users/rohangavankar/PycharmProjects/test_app/bank/bank-full.csv")
X = dataset.loc[:, ["age","marital_no","education_no","balance","poutcome_no",'duration']].values
y = dataset.loc[:, ['y_no']].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Feature Scaling

##for data train comment out scale and then show it to show difference in accuracy
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

log = LogisticRegression(random_state = 0)
log.fit(X_train, y_train)

# Predicting the Test set results
y_pred = log.predict(X_test)
print(X_test, y_pred)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy is:", round(accuracy_score(y_test, y_pred), 4))


# Predicting the Test set results
y_pred = log.predict(X_test)
importance = log.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))



tn, fp, fn, tp = cm[0][0],cm[0][1] , cm[1][0], cm[1][1]
print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
print("Accuracy is:", round(accuracy_score(y_test, y_pred), 4))
recall = tp/(tp+fn)
precision =  tp/(tp+fp)
print("Recall is:" , recall)
print("Precision is:" , precision)
print('F-Measure is:',  2*((recall*precision)/(recall+precision)))
