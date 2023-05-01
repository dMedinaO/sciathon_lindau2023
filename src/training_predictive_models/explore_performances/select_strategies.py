import pandas as pd
import sys
import numpy as np
import json

def select_models_by_performances(df_data, value, column):
    filter_df = df_data.loc[df_data[column] <= value]
    return filter_df

def create_overfitting_rate_column(df_data, train_column, test_column):

    array_value = 1 - df_data[train_column]*df_data[test_column]
    return array_value

columns_performance = ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'train_f1_weighted', 'train_recall_weighted', 'train_precision_weighted', 'train_accuracy']
columns_config = ['iteration', 'description', 'is_scale','encoding_smile', 'encoding_protein', 'concat_strategy']
columns_rate = ['overfitting_rate_accuracy', 'overfitting_rate_f1', 'overfitting_rate_precision', 'overfitting_rate_recall']

print("Get console params")
df_data = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]

print("Estimating overfitting rate")
#create column with overfitting rate 1 - training*test
overfitting_rate_accuracy = create_overfitting_rate_column(df_data, 'train_accuracy', 'test_accuracy')
overfitting_rate_f1 = create_overfitting_rate_column(df_data, 'train_f1_weighted', 'test_f1_score')
overfitting_rate_precision = create_overfitting_rate_column(df_data, 'train_precision_weighted', 'test_precision')
overfitting_rate_recall = create_overfitting_rate_column(df_data, 'train_recall_weighted', 'test_recall')

df_data['overfitting_rate_accuracy'] = overfitting_rate_accuracy
df_data['overfitting_rate_f1'] = overfitting_rate_f1
df_data['overfitting_rate_precision'] = overfitting_rate_precision
df_data['overfitting_rate_recall'] = overfitting_rate_recall

#get average of explore models, this is by encoding_smile, encoding_protein, concat_strategy, is_scale, and description
print("Average columns using group by")

df_summary = df_data.groupby(by=['description', 'encoding_smile', 'encoding_protein', 'concat_strategy', 'is_scale']).mean().reset_index()
df_summary.to_csv("{}data_group_by.csv".format(path_export), index=False)

print("Get best performances using the overfitting rate")
min_rate_accuracy = np.min(df_summary['overfitting_rate_accuracy'])
min_rate_f1 = np.min(df_summary['overfitting_rate_f1'])
min_rate_precision = np.min(df_summary['overfitting_rate_precision'])
min_rate_recall = np.min(df_summary['overfitting_rate_recall'])

filter_accuracy = select_models_by_performances(df_summary, min_rate_accuracy, 'overfitting_rate_accuracy')
filter_f1 = select_models_by_performances(df_summary, min_rate_f1, 'overfitting_rate_f1')
filter_precision = select_models_by_performances(df_summary, min_rate_precision, 'overfitting_rate_precision')
filter_recall = select_models_by_performances(df_summary, min_rate_recall, 'overfitting_rate_recall')

df_full = pd.concat([filter_accuracy, filter_f1, filter_recall, filter_precision], axis=0)

df_full.to_csv("{}filter_data.csv".format(path_export), index=False)