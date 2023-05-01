from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from rdkit import Chem 
import pandas as pd
import numpy as np

class smile_encoder(object):

    def __init__(
        self, 
        smiles_df=None, 
        pretrained_model=None):

        self.smiles_df = smiles_df
        self.pretrained_model = pretrained_model
    
    def encoding_smiles_using_mol2vec(
        self, 
        column_name=None):
        
        #instance a model
        model = word2vec.Word2Vec.load(
            self.pretrained_model)

        #create a response df
        df_response = pd.DataFrame()
        df_response[column_name] = self.smiles_df[column_name]

        #create mol structure from the smile
        df_response['mol'] = self.smiles_df[column_name].apply(lambda x: Chem.MolFromSmiles(x))

        #create sentences and process response
        df_response['sentence'] = df_response.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
        df_response['mol2vec'] = [DfVec(x) for x in sentences2vec(df_response['sentence'], model, unseen='UNK')]
        
        #encoding
        encoding_response = np.array([x.vec for x in df_response['mol2vec']])

        #create a df with the encoding
        header = ["p_{}".format(i) for i in range(len(encoding_response[0]))]
        df_encoding = pd.DataFrame(encoding_response, columns=header)
        df_encoding[column_name] = self.smiles_df[column_name]

        df_response = df_response.merge(df_encoding, left_on=column_name, right_on=column_name)
        df_response = df_response.drop(columns=['sentence', 'mol2vec', 'mol'])
        return df_response
