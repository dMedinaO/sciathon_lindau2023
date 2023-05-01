import pandas as pd
import sys
import os

path_input = sys.argv[1]
encoding_smile = sys.argv[2]
encoding_protein = sys.argv[3]
concat_strategy = sys.argv[4]
name_export = sys.argv[5]

suffix_to_concat = sys.argv[6]

list_df_scale = []
list_df_not_scale = []

list_csv = os.listdir(path_input)

for document in list_csv:
    df = pd.read_csv("{}{}".format(path_input, document))

    if suffix_to_concat in document:
        list_df_not_scale.append(df)
    else:
        list_df_scale.append(df)

#concat
df_scale = pd.concat(list_df_scale, axis=0)
df_not_scale = pd.concat(list_df_not_scale, axis=0)

#change index
df_scale = df_scale.reset_index()
df_not_scale = df_not_scale.reset_index()

df_scale['is_scale'] = 1
df_not_scale['is_scale'] = 0

df_full = pd.concat([df_scale, df_not_scale], axis=0)
df_full = df_full.reset_index()

df_full['encoding_smile'] = encoding_smile
df_full['encoding_protein'] = encoding_protein
df_full['concat_strategy'] = concat_strategy

df_full = df_full.drop(columns=['level_0', 'index'])
df_full.to_csv(name_export, index=False)
