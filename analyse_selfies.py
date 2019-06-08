'''
Written by Jan H. Jensen 2019
'''

from selfies import encoder, decoder
import pandas as pd
from rdkit import Chem

pd.set_option('max_colwidth',200)
df = pd.read_csv('ZINC_250k.smi', sep=" ", header=None)
df.columns = ["smiles"]

rows = 1000
symbols_list = []
for index, row in df.iterrows():
    smiles = row['smiles']
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    smiles = Chem.MolToSmiles(mol)
    symbols = encoder(smiles).split('][')
    for symbol in symbols:
        symbol = symbol.replace(']','').replace('[','')
        if symbol not in symbols_list:
            symbols_list.append(symbol)

    
print(symbols_list)