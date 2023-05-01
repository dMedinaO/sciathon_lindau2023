from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import random

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
dataset = pd.read_csv(sys.argv[1])
name_export = sys.argv[2]

#create a balanced dataset
print("Creating balanced dataset")
dataset, responses = create_balanced_dataset(dataset, 'response_activity')

X_train, X_test, y_train, y_test = train_test_split(dataset, responses, train_size=0.80, test_size=0.20, random_state=42)

tpot = TPOTClassifier(generations=50, population_size=50, verbosity=2, random_state=42, n_jobs=-1)
tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
tpot.export(name_export)
