import pandas as pd
import sys
import numpy as np
import json

def get_average_by_algorithm(df_filter, list_performances):
    row_response = []
    for k in range(len(list_performances)):
        value = np.mean(df_filter[list_performances[k]])
        row_response.append(value)

    return row_response

def add_overfitting_rate(column_train, columnt_test, dataset):

    response = 1- dataset[column_train]*dataset[columnt_test]
    return response

df_performances = pd.read_csv(sys.argv[1])
path_to_export = sys.argv[2]

#add overfitting rate
columns_performances = ['test_accuracy','test_f1_score','test_precision','test_recall', 'train_f1_weighted','train_recall_weighted','train_precision_weighted','train_accuracy', 'over_rate_acuracy', 'over_rate_f1', 'over_rate_precision', 'over_rate_recall']

print("Get overfitting rate")
df_performances['over_rate_acuracy'] = add_overfitting_rate('test_accuracy', 'train_accuracy', df_performances)
df_performances['over_rate_f1'] = add_overfitting_rate('test_f1_score', 'train_f1_weighted', df_performances)
df_performances['over_rate_precision'] = add_overfitting_rate('test_precision', 'train_precision_weighted', df_performances)
df_performances['over_rate_recall'] = add_overfitting_rate('test_recall', 'train_recall_weighted', df_performances)

print("Exporting performances with overfitting rate")
name_export = "{}1_df_with_overfitting_rate.csv".format(path_to_export)
df_performances.to_csv(name_export, index=False)

#get average performances by group and algorithm
unique_algorithm = df_performances['description'].unique()

print("Get average by algorithm and by group")
matrix_data = []
for i in range(8):
    name_group = "group_{}".format(i)
    print("Processing group: ", name_group)

    filter_df = df_performances.loc[df_performances['id_group'] == name_group]
    filter_df.reset_index(inplace=True)

    for j in range(len(unique_algorithm)):
        print("Processing algorithm: ", unique_algorithm[j])
        df_filter = filter_df.loc[filter_df['description'] == unique_algorithm[j]]
        response_average = get_average_by_algorithm(df_filter, columns_performances)
        response_average.insert(0, unique_algorithm[j])
        response_average.insert(0, name_group)
        matrix_data.append(response_average)
        
header = columns_performances
header.insert(0, "algorithm")
header.insert(0, "group_id")
df_data_average = pd.DataFrame(matrix_data, columns=header)

print("Exporting average df")
name_export = "{}2_df_with_average_performances.csv".format(path_to_export)
df_data_average.to_csv(name_export, index=False)

print("Get best method by group and by performance")
dict_summary = {}
for i in range(8):
    name_group = "group_{}".format(i)
    filter_df = df_data_average.loc[df_data_average['group_id'] == name_group]
    filter_df.reset_index(inplace=True)
    dict_data = {}
    for performance in columns_performances:

        if performance not in ["algorithm", "group_id"]:
            dict_performance = {}
            list_description = []
            if "over_" not in performance:
                max_value = np.max(filter_df[performance])
                df_with_performances = filter_df.loc[filter_df[performance]>=max_value]
                unique_algorithm = df_with_performances['algorithm'].unique()
                list_description = [value for value in unique_algorithm]
                dict_performance.update({"value_performance": max_value})
            else:
                min_value = np.min(filter_df[performance])
                df_with_performances = filter_df.loc[filter_df[performance]<=min_value]
                unique_algorithm = df_with_performances['algorithm'].unique()
                list_description = [value for value in unique_algorithm]
                dict_performance.update({"value_performance": min_value})
            dict_performance.update({"list_description":list_description})
            dict_data.update({performance: dict_performance})
    dict_summary.update({name_group:dict_data})

print("Export results")
name_export = "{}summary_results_best_performances.json".format(path_to_export)
with open(name_export, 'w') as doc_export:
    json.dump(dict_summary, doc_export)
