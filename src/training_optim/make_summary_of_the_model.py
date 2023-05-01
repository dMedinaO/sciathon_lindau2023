import sys
import random
import numpy as np
import pandas as pd

#to train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#console params
dataset = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]

#create a balanced dataset
responses = dataset['response_activity']
dataset = dataset.drop(columns=['response_activity'])

X_train, X_test, y_train, y_test = train_test_split(dataset, responses, train_size=0.80, test_size=0.20, random_state=42)

clf = RandomForestClassifier(criterion="gini", max_depth=340, max_features="sqrt", min_samples_leaf=1, min_samples_split=2, n_estimators=50)
clf.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

y_pred = clf.predict(X_test)

#confusion matrix
color = 'white'
matrix = plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.savefig("{}confusion_matrix.png".format(path_export))

plt.clf()

#roc curve
y_score1 = clf.predict_proba(X_test)[:,1]

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("{}roc_curve.png".format(path_export))

print('roc_auc_score: ', roc_auc_score(y_test, y_score1))
