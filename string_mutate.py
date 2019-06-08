'''
Written by Emilie S. Henault and Jan H. Jensen 2019 
'''
from rdkit import Chem
from rdkit.Chem import AllChem

import random
import numpy as np

import crossover as co
import string_crossover as stco

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def get_symbols():
    if stco.string_type == 'smiles':
        symbols = ['C', 'O', '(', '=', ')', '[C@@H]', '[C@H]', 'H', '1', 'N', '2', '3', 'F', 'S', 
                    'Cl', '#', '+', '-', '/', '4', 'B', 'Br', '\\', '5', 'I']
    
    if stco.string_type == 'deepsmiles':
        symbols = ['C', 'N', ')', 'S', '=', 'O', '6', '5', '9', 'B', 'Br', '[C@@H]', '[C@H]', 'H', 
                    '+', '%10', '%11', '%12', '%13', '%14', '%15', '%16', '%17', '1', '0', '2', 'l',
                    '3', 'F', '#', '7', 'I', '-', '/', '\\', '4', '8']
    
    if stco.string_type == 'selfies':
        symbols = ['C', 'Branch1_2', 'epsilon', 'Branch1_3', '=C', 'O', '#N', '=O', 'N', 'Ring1', 
               'Branch1_1', 'F', '=N', '#C', 'C@@H', 'S', 'Branch2_2', 'Ring2', 'Branch2_3', 
               'Branch2_1', 'Cl', 'O-', 'C@H', 'NH+', 'C@', 'Br', '/C', '/O', 'NH3+', '=S', 'NH2+', 
               'C@@', '=N+', '=NH+', 'N+', '\\C', '\\O', '/N', '/S', '\\S', 'S@', '\\O-', 'N-', '/NH+', 
               'S@@', '=NH2+', '/O-', 'S-', '/S-', 'I', '\\N', '\\Cl', '=P', '/F', '/C@H', '=OH+', 
                '\\S-', '=S@@', '/C@@H', 'P', '=S@', '\\C@@H', '/S@', '/Cl', '=N-', '/N+', 'NH-', 
                '\\C@H', 'P@@H', 'P@@', '\\N-', 'Expl\\Ring1', '=P@@', '=PH2', '#N+', '\\NH+', 'P@', 
                'P+', '\\N+', 'Expl/Ring1', 'S+', '=O+', '/N-', 'CH2-', '=P@', '=SH+', 'CH-', '/Br', 
                '/C@@', '\\Br', '/C@', '/O+', '\\F', '=S+', 'PH+', '\\NH2+', 'PH', '/NH-', '\\S@', 'S@@+', 
                '/NH2+', '\\I']

    return symbols


def mutate(mol,mutation_rate):
    if random.random() > mutation_rate:
        return mol
    Chem.Kekulize(mol,clearAromaticFlags=True)
    child = stco.mol2string(mol)
    symbols = get_symbols()
    for i in range(50):
        random_number = random.random()
        mutated_gene = random.randint(0, len(child) - 1)
        random_symbol_number = random.randint(0, len(symbols)-1)
        new_child = list(child)
        random_number = random.random()
        new_child[mutated_gene] = symbols[random_symbol_number]
        new_child_mol = stco.string2mol(new_child)
        #print(child_smiles,Chem.MolToSmiles(child_mol),child_mol,co.mol_OK(child_mol))
        if co.mol_OK(new_child_mol):
            return new_child_mol

    return mol

if __name__ == "__main__":
    co.average_size = 39.15
    co.size_stdev = 3.50
    mutation_rate = 1.0
    stco.string_type = 'smiles'
    mol = Chem.MolFromSmiles('CCC(CCCC)C')
    child = mutate(mol,mutation_rate)
    if child:
        print(Chem.MolToSmiles(child))
    else:
        print(child)
