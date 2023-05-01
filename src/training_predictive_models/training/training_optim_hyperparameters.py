from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import random
import numpy as np

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

dataset = open_doc.readline().replace("\n", "")
name_export = open_doc.readline().replace("\n", "")

open_doc.close()


#create a balanced dataset
print("Creating balanced dataset")
dataset, responses = create_balanced_dataset(dataset, 'response_activity')

X_train, X_test, y_train, y_test = train_test_split(dataset, responses, train_size=0.80, test_size=0.20, random_state=42)

n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 1)]
max_features = ['auto', 'sqrt','log2']
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 20, num = 1)]
min_samples_leaf = [int(x) for x in np.linspace(start = 1, stop = 20, num = 1)]
criterion_list = ['gini', 'entropy', 'log_loss']

param = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'criterion':criterion_list}

tpot_classifier = TPOTClassifier(generations= 50, population_size= 50, verbosity= 2,
                                 config_dict={'sklearn.ensemble.RandomForestClassifier': param}, 
                                 cv = 10, scoring = 'accuracy', n_jobs=-1)

tpot_classifier.fit(X_train,y_train)

tpot_classifier.export(name_export)