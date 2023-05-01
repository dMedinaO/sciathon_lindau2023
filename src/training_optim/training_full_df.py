import sys
import random
import numpy as np
import pandas as pd

#to train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#to get performances
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#console params
doc_config = sys.argv[1]

open_doc = open(doc_config, 'r')

dataset = pd.read_csv(open_doc.readline().replace("\n", ""))
name_export = open_doc.readline().replace("\n", "")
iteration = int(open_doc.readline().replace("\n", ""))

open_doc.close()

#create a balanced dataset
responses = dataset['response_activity']
dataset = dataset.drop(columns=['response_activity'])

X_train, X_test, y_train, y_test = train_test_split(dataset, responses, train_size=0.80, test_size=0.20, random_state=42)

clf = RandomForestClassifier(criterion="gini", max_depth=340, max_features="sqrt", min_samples_leaf=1, min_samples_split=2, n_estimators=50)
clf.fit(X_train, y_train)

print("Make predictions")
responses_prediction = clf.predict(X_test)

print("Get performances")
accuracy_value = accuracy_score(y_test, responses_prediction)
f1_score_value = f1_score(y_test, responses_prediction, average='weighted')
precision_values = precision_score(y_test, responses_prediction, average='weighted')
recall_values = recall_score(y_test, responses_prediction, average='weighted')

print("Export results")
row = [iteration, accuracy_value, precision_values, recall_values, f1_score_value]
matrix_data = [row]

df_export = pd.DataFrame(matrix_data, columns=['iteration', 'accuracy_value', 'precision_values', 'recall_values', 'f1_score_value'])
df_export.to_csv(name_export, index=False)