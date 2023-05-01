import pandas as pd
import sys

#datasets
encoding_seq = pd.read_csv(sys.argv[1])
encoding_smiles = pd.read_csv(sys.argv[2])
full_dataset = pd.read_csv(sys.argv[3])

#name to save the results
name_export = sys.argv[4]

#merge smiles with encoders
data_merge = full_dataset.merge(encoding_smiles, left_on='SMILES', right_on='SMILE')

#merge enzyme with encoders
data_merge = data_merge.merge(encoding_seq, left_on='enzyme_reaction', right_on='id_seq')

#remove columns
data_merge.drop(columns=['Name', 'SMILES', 'SMILE', 'enzyme_reaction', 'id_seq'], inplace=True)

data_merge.to_csv(name_export, index=False)

