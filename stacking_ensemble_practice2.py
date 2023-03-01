from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from bankproject.ProjectBank import do_etl
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# get the dataset
def get_dataset():
#    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
    df = do_etl("/Users/rohangavankar/PycharmProjects/test_app/bank/bank.csv")
    #df = df.iloc[0:2000,:]
    X = df.loc[:, ['age', 'job_no', 'marital_no', 'education_no', 'default_no', 'balance', 'housing_no', 'loan_no', 'contact_no', 'day', 'month_no', \
                        'duration', 'campaign', 'pdays', 'previous', 'poutcome_no']].values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    #X_test = sc.transform(X_test)
    y = df.loc[:, 'y_no'].values
    return X, y


# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    #level0.append(('lr', LogisticRegression()))
    #level0.append(('knn', KNeighborsClassifier()))
    #level0.append(('cart', DecisionTreeClassifier()))
    #level0.append(('AB', AdaBoostClassifier(n_estimators=100, random_state=0)))
    level0.append(( 'GBM', GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=1, random_state=0)))
    #level0.append(('svm', SVC()))
    #level0.append(('bayes', GaussianNB()))
    level0.append(('xgb', xgb.XGBClassifier(n_jobs=1)))
    level0.append(('RF', RandomForestClassifier()))    # define meta learner model
    #level1 = LogisticRegression()
    #level1 = xgb.XGBClassifier(n_jobs=1)
    level1 = RandomForestClassifier()

    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model




# define dataset
X, y = get_dataset()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# get the models to evaluate
classifier = get_stacking()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
tn, fp, fn, tp = cm[0][0],cm[0][1] , cm[1][0], cm[1][1]
print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
print("Accuracy is:", round(accuracy_score(y_test, y_pred), 4))
recall = tp/(tp+fn)
precision =  tp/(tp+fp)
print("Recall is:" , recall)
print("Precesion is:" , precision)
print('F-Measure is:',  2*((recall*precision)/(recall+precision)))