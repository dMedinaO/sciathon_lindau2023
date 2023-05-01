import pandas as pd
import sys
import numpy as np

def create_overfitting_rate_column(df_data, train_column, test_column):

    array_value = 1 - df_data[train_column]*df_data[test_column]
    return array_value

def select_models_by_performances(df_data, value, column):
    filter_df = df_data.loc[df_data[column] <= value]
    return filter_df

df_data = pd.read_csv(sys.argv[1])

columns = ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'train_f1_weighted', 'train_recall_weighted', 'train_precision_weighted', 'train_accuracy']

#create summary dictionaries
dict_max = {}
dict_min = {}
dict_average = {}

for column in columns:
    max_value = np.max(df_data[column])
    min_value = np.min(df_data[column])
    average_value = np.mean(df_data[column])

    dict_average.update({column: average_value})
    dict_max.update({column: max_value})
    dict_min.update({column: min_value})

#create column with overfitting rate 1 - training*test
overfitting_rate_accuracy = create_overfitting_rate_column(df_data, 'train_accuracy', 'test_accuracy')
overfitting_rate_f1 = create_overfitting_rate_column(df_data, 'train_f1_weighted', 'test_f1_score')
overfitting_rate_precision = create_overfitting_rate_column(df_data, 'train_precision_weighted', 'test_precision')
overfitting_rate_recall = create_overfitting_rate_column(df_data, 'train_recall_weighted', 'test_recall')

df_data['overfitting_rate_accuracy'] = overfitting_rate_accuracy
df_data['overfitting_rate_f1'] = overfitting_rate_f1
df_data['overfitting_rate_precision'] = overfitting_rate_precision
df_data['overfitting_rate_recall'] = overfitting_rate_recall

#we will select using the overfitting rates
min_rate_accuracy = np.min(df_data['overfitting_rate_accuracy'])
min_rate_f1 = np.min(df_data['overfitting_rate_f1'])
min_rate_precision = np.min(df_data['overfitting_rate_precision'])
min_rate_recall = np.min(df_data['overfitting_rate_recall'])

filter_accuracy = select_models_by_performances(df_data, min_rate_accuracy, 'overfitting_rate_accuracy')
filter_f1 = select_models_by_performances(df_data, min_rate_f1, 'overfitting_rate_f1')
filter_precision = select_models_by_performances(df_data, min_rate_precision, 'overfitting_rate_precision')
filter_recall = select_models_by_performances(df_data, min_rate_recall, 'overfitting_rate_recall')

filter_accuracy['filter_by'] = "best rate accuracy"
filter_f1['filter_by'] = "best rate f1"
filter_recall['filter_by'] = "best rate recall"
filter_precision['filter_by'] = "best rate precision"

df_full = pd.concat([filter_accuracy, filter_f1, filter_recall, filter_precision], axis=0)

df_full = df_full.drop(columns=['level_0', 'index'])
df_full.to_csv(sys.argv[2]+"filter_data.csv", index=False)