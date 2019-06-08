'''
Written by Emilie S. Henault and Jan H. Jensen 2019 
'''
from rdkit import Chem
from rdkit.Chem import AllChem

import random
import numpy as np

import crossover as co

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import deepsmiles
converter = deepsmiles.Converter(rings=True, branches=True)
from selfies import encoder, decoder

def cut_point(parent):
  m = random.randint(0, len(parent) - 1)
  return m

def mol2string(mol):
    Chem.Kekulize(mol, clearAromaticFlags=True)
    smiles = Chem.MolToSmiles(mol)

    if selfies:
        return encoder(smiles).split('][')

    if d_smiles:
        string = converter.encode(smiles)

    return list(string)

def string2mol(string):
    if selfies:
        string = ']['.join(string)
        try:
            smiles = decoder(string,PrintErrorMessage=False)
        except:
            return None
    else:
        string = ''.join(string)

    if d_smiles:
        try:
            smiles = converter.decode(string)
        except deepsmiles.DecodeError as e:
            return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def crossover(parent_a_mol,parent_b_mol):
    parent_a, parent_b = mol2string(parent_a_mol), mol2string(parent_b_mol)

    for _ in range(50):
        cut_point_a = cut_point(parent_a)
        cut_point_b = cut_point(parent_b)
        a1 = parent_a[0:cut_point_a]
        b2 = parent_b[cut_point_b:len(parent_b)]
        child_string = a1 + b2
        child_mol = string2mol(child_string)
        #print(child_smiles,Chem.MolToSmiles(child_mol),child_mol,co.mol_OK(child_mol))
        if co.mol_OK(child_mol):
            return child_mol

    return None

if __name__ == "__main__":
    co.average_size = 39.15
    co.size_stdev = 3.50
    d_smiles = False
    selfies = True
    mol1 = Chem.MolFromSmiles('CCC(CCCC)C')
    mol2 = Chem.MolFromSmiles('OCCCCCCO')
    mol2 = Chem.MolFromSmiles('OCCCCCCc1ccccc1')
    child = crossover(mol1,mol2)
    if(child):
        print(Chem.MolToSmiles(child))
    else:
        print(child)