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

#function to create a balanced dataset class
def create_balanced_dataset(dataset, column_response):
    print("Split dataset")
    positive_class = dataset.loc[dataset[column_response] == True]
    negative_class = dataset.loc[dataset[column_response] == False]

    #process the negative class 
    index_values = [index for index in negative_class.index]
    random.shuffle(index_values)
    index_values = index_values[:len(positive_class)] # balanced with the lenght of positive values

    negative_class['index'] = [index for index in negative_class.index]
    negative_class['is_in_select'] = negative_class['index'].isin(index_values)
    negative_class_filter = negative_class.loc[negative_class['is_in_select'] == True]
    negative_class_filter = negative_class_filter.drop(columns=['index', 'is_in_select', 'response_activity'])
    negative_class_filter = negative_class_filter.reset_index()

    #remove column with activity for positive class
    positive_class = positive_class.drop(columns=['response_activity'])
    positive_class = positive_class.reset_index()

    #create column response
    positive_values = [1 for i in range(len(positive_class))]
    negative_values = [0 for i in range(len(negative_class_filter))]

    #concat the dataset
    full_df = pd.concat([positive_class, negative_class_filter], axis=0)
    full_df = full_df.drop(columns=['index'])
    responses = positive_values + negative_values

    return full_df, responses

#console params
doc_config = sys.argv[1]
open_doc = open(doc_config, 'r')

dataset = pd.read_csv(open_doc.readline().replace("\n", ""))
name_export = open_doc.readline().replace("\n", "")
iteration = int(open_doc.readline().replace("\n", ""))

open_doc.close()

#create a balanced dataset
print("Creating balanced dataset")
dataset, responses = create_balanced_dataset(dataset, 'response_activity')

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