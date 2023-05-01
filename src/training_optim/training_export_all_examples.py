import sys
import random
import numpy as np
import pandas as pd

#to train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#to export model
from joblib import dump

#console params
dataset = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]

#create a balanced dataset
responses = dataset['response_activity']
dataset = dataset.drop(columns=['response_activity'])

X_train, X_test, y_train, y_test = train_test_split(dataset, responses, train_size=0.80, test_size=0.20, random_state=42)

clf = RandomForestClassifier(criterion="gini", max_depth=340, max_features="sqrt", min_samples_leaf=1, min_samples_split=2, n_estimators=50)
clf.fit(X_train, y_train)

print("Export results")
dump(clf, "{}model_optim_with_all_examples.joblib".format(path_export))