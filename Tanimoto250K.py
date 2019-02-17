'''
Written by Jan H. Jensen 2018
'''
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

import pandas as pd

import scoring_functions as sc



Celecoxib = 'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
target = Chem.MolFromSmiles(Celecoxib)
fp_target = sc.get_ECFP4(target)

pd.set_option('max_colwidth',200)
df = pd.read_csv('ZINC_250k.smi', sep=" ", header=None)
df.columns = ["smiles"]

rows = 1000
scores = []
for index, row in df.iloc[0:rows].iterrows():
	smiles = row['smiles']
	#print(smiles)
	mol = Chem.MolFromSmiles(smiles)
	#print(mol,Chem.MolToSmiles(mol))
	fp_mol = sc.get_ECFP4(mol)
	score = TanimotoSimilarity(fp_mol,fp_target)
	#print(score)
	scores.append(score)

df2 = df.iloc[0:rows]
#print(df2)
#print(pd.DataFrame(scores, columns=['scores']))

df3 = pd.DataFrame(scores, columns=['score'])

df2 = df2.join(df3['score'])

df2.sort_values(by=['score'], ascending=False, inplace=True)

#print(df2)

subset1 = df2[df2.score <= 0.323]
#print(subset1.head(100))
#subset2 = subset1[subset1.score > 0.2]

#print(subset2)


for index, row in subset1.iloc[0:100].iterrows():
	smiles = row['smiles']
	score = row['score']
	#print(smiles,score)
	print(smiles)
		
