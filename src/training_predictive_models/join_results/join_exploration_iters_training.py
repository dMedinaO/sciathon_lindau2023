import pandas as pd
import sys
import os

path_input = sys.argv[1]
doc_list = os.listdir(path_input)
doc_list = [value for value in doc_list if "." in value]

df_list_full = []

for i in range(8):
    print("Processing group: ", i)
    df_list = []

    for document in doc_list:
        if "group{}".format(i) in document:
            name_doc = "{}{}".format(path_input, document)
            df_read = pd.read_csv(name_doc)
            df_list.append(df_read)
    
    print("Join group dataset")
    df_full_grupo = pd.concat(df_list, axis=0)
    df_full_grupo['id_group'] = "group_{}".format(i)
    df_list_full.append(df_full_grupo)

print("Join the datasets")
df_full = pd.concat(df_list_full, axis=0)

print("Export csv full")
name_export = "{}full_exploring_training.csv".format(path_input)
df_full.to_csv(name_export, index=False)

print("Move results to run_all")
for document in doc_list:
    command = "mv {}{} {}run_all/".format(path_input, document, path_input)
    print(command)
    os.system(command)

    