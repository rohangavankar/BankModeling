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


# Split the train and test samples
print(len(x))
test_samples = 1000
x_train, y_train = x[:-test_samples], y[:-test_samples]
x_test, y_test = x[-test_samples:], y[-test_samples:]

import joblib as jl
model_1=jl.load('/Users/rohangavankar/PycharmProjects/test_app/Models/votingModel1.joblib')
model_2=jl.load('/Users/rohangavankar/PycharmProjects/test_app/Models/votingModel2.joblib')
model_3=jl.load('/Users/rohangavankar/PycharmProjects/test_app/Models/votingModel3.joblib')

# predicts the classes of the test data
predictions_1 = model_1.predict(x_test)
predictions_2 = model_2.predict(x_test)
predictions_3 = model_3.predict(x_test)


# Combine the predictions with hard voting
vote_predictions = []
# Iterate for each predicted sample
for i in range(test_samples):
    #count[0] is vote count for 0 value
    #count[1] is vote count for 1 value
    counts = [0 for _ in range(2)]
    print(y_test[i], predictions_1[i],predictions_2[i],predictions_3[i])
    counts[predictions_1[i]] = counts[predictions_1[i]]+1
    counts[predictions_2[i]] = counts[predictions_2[i]]+1
    counts[predictions_3[i]] = counts[predictions_3[i]]+1
    print(counts)
    
    final = argmax(counts)  # class/Label with max votes
    
    vote_predictions.append(final)
    print(y_test[i], predictions_1[i], predictions_2[i], predictions_3[i], vote_predictions[i])

print('M1:', accuracy_score(y_test, predictions_1))
print('M2:', accuracy_score(y_test, predictions_2))
print('M3:', accuracy_score(y_test, predictions_3))
# Accuracy of hard voting
print('-'*30)
print('Hard Voting:', accuracy_score(y_test, vote_predictions))