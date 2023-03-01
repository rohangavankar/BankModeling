# --- SECTION 1 ---
# Libraries and data loading

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np

dataset = pd.read_csv("/Users/rohangavankar/PycharmProjects/test_app/data/Ads-2.csv")

train_size = 300
x_train = dataset.iloc[:train_size, [2, 3]].values
y_train = dataset.iloc[:train_size, 4].values

x_test = dataset.iloc[train_size:, [2, 3]].values
y_test = dataset.iloc[train_size:, 4].values

# Create bootstrap samples and train the classifiers with bootstrap samples

ensemble_size = 15
base_models = []

for _ in range(ensemble_size):
 bootstrap_sample_indices = np.random.randint(0, train_size, size=train_size)
 bootstrap_x = x_train[bootstrap_sample_indices]
 bootstrap_y = y_train[bootstrap_sample_indices]
 decision_tree = DecisionTreeClassifier()
 decision_tree.fit(bootstrap_x, bootstrap_y)
 base_models.append(decision_tree)

# Predict with the base models and evaluate them
base_predictions = []
base_accuracy = []
for model in base_models:
 predictions = model.predict(x_test)
 base_predictions.append(predictions)
 model_acc = metrics.accuracy_score(y_test, predictions)
 base_accuracy.append(model_acc)

# Combine the base models' predictions

ensemble_predictions = []
# Find the most voted class for each test instance
for i in range(len(y_test)):
    counts = [0 for _ in range(10)]
    for model_predictions in base_predictions:
        counts[model_predictions[i]] = counts[model_predictions[i]] + 1
    # Find the class with most votes
    final = np.argmax(counts)
    # Add the class to the final predictions
    ensemble_predictions.append(final)

bagg_ensemble_acc = metrics.accuracy_score(y_test, ensemble_predictions)

# Print the model accuracies
print('Base models:')
print('#'*80)
for index, acc in enumerate(sorted(base_accuracy)):
 print(f'model {index+1} Accuracy: %.4f' % acc)
print('#'*80)
print('Bagging model Accuracy: %.4f' % bagg_ensemble_acc)