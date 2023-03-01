# Import the required libraries
# Import the required libraries
from sklearn import  svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from numpy import argmax
from sklearn.preprocessing import StandardScaler

from bankproject.ProjectBank import do_etl
# get the dataset
def get_dataset():
    df = do_etl("/Users/rohangavankar/PycharmProjects/test_app/bank/bank.csv")
    X = df.loc[:, ['age', 'job_no', 'marital_no', 'education_no', 'default_no', 'balance', 'housing_no', 'loan_no', 'contact_no', 'day', 'month_no', \
                        'duration', 'campaign', 'pdays', 'previous', 'poutcome_no']].values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = df.loc[:, 'y_no'].values
    return X, y

# Load the dataset
x, y = get_dataset()

model_1 = LogisticRegression()
model_2 = DecisionTreeClassifier()
model_3 = svm.SVC(gamma=0.001)

# Split the train and test samples
print(len(x))
test_samples = 1000
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

# Fit models with the train data
model_1.fit(x_train, y_train)
model_2.fit(x_train, y_train)
model_3.fit(x_train, y_train)

import joblib as jl
jl.dump(model_1, "/Users/rohangavankar/PycharmProjects/test_app/Models/votingModel1.joblib")
jl.dump(model_2, "/Users/rohangavankar/PycharmProjects/test_app/Models/votingModel2.joblib")
jl.dump(model_2, "/Users/rohangavankar/PycharmProjects/test_app/Models/votingModel3.joblib")


