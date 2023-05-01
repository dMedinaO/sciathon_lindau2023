import pandas as pd
import sys
import os

path_to_use = sys.argv[1]

list_csv = os.listdir(path_to_use)

df_list = []

for element in list_csv:
    df = pd.read_csv("{}{}".format(path_to_use, element))
    df_list.append(df)

df_full = pd.concat(df_list, axis=0)
df_full.reset_index(inplace=True)
df_full.to_csv("{}full_explore_strategies.csv".format(path_to_use), index=False)

for element in list_csv:
    command = "rm {}{}".format(path_to_use, element)
    print(command)
    os.system(command)