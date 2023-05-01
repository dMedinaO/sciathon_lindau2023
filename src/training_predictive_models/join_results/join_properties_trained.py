import pandas as pd
import sys
import os

path_input = sys.argv[1]
encoding_smile = sys.argv[2]
concat_strategy = sys.argv[3]
is_scale = int(sys.argv[4])
name_export = sys.argv[5]

list_document = os.listdir(path_input)

list_df = []

for document in list_document:

    df_read = pd.read_csv("{}{}".format(path_input, document))

    #search encoding strategy
    encoding_protein = document.split("_")[0]
    if "FFT" in document:
        encoding_protein = encoding_protein+"_FFT"
    
    df_read['is_scale'] = is_scale
    df_read['encoding_smile'] = encoding_smile
    df_read['encoding_protein'] = encoding_protein
    df_read['concat_strategy'] = concat_strategy

    list_df.append(df_read)

full_df = pd.concat(list_df, axis=0)
full_df = full_df.reset_index()
full_df = full_df.drop(columns="index")

full_df.to_csv(name_export, index=False)
