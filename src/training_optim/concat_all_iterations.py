import pandas as pd
import os
import sys

path_input = sys.argv[1]

list_doc = os.listdir(path_input)

list_df = []

for doc in list_doc:
    if doc != "full_training.csv" and ".joblib" not in doc:
        df = pd.read_csv("{}{}".format(path_input, doc))
        list_df.append(df)

df_full = pd.concat(list_df, axis=0)

for doc in list_doc:
    if doc != "full_training.csv" and ".joblib" not in doc:
        command = "rm {}{}".format(path_input, doc)
        os.system(command)

df_full.to_csv("{}full_training_with_all_examples.csv".format(path_input), index=False)
