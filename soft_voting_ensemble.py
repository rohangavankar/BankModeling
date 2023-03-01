# Non-Generative Soft voting example
# Import the required libraries
from sklearn import  svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from numpy import argmax
from sklearn.preprocessing import StandardScaler
from bankproject.ProjectBank import do_etl
from sklearn.ensemble import VotingClassifier
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
model_3 = svm.SVC(gamma=0.001,probability=True)
# Split the train and test samples,
print(len(x))
test_samples = 1000
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]
voting = VotingClassifier([('LR', model_1),
                           ('DT', model_2),
                           ('SVM', model_3)],
                            voting='soft')
voting.fit(x_train, y_train)
# Fit models with the train data
model_1.fit(x_train, y_train)
model_2.fit(x_train, y_train)
model_3.fit(x_train, y_train)
# predicts the classes of the test data
predictions_1 = model_1.predict(x_test)
predictions_2 = model_2.predict(x_test)
predictions_3 = model_3.predict(x_test)
soft_vote_predictions = voting.predict(x_test)
print('M1:', accuracy_score(y_test, predictions_1))
print('M2:', accuracy_score(y_test, predictions_2))
print('M3:', accuracy_score(y_test, predictions_3))
# Accuracy of hard voting
print('-'*30)
print('Soft Voting:', accuracy_score(y_test, soft_vote_predictions))
