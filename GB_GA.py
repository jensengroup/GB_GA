'''
Written by Jan H. Jensen 2018
'''
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import numpy as np
import random
import time
import sys

import crossover as co
import mutate as mu
import scoring_functions as sc

def read_file(file_name):
  mol_list = []
  with open(file_name,'r') as file:
    for smiles in file:
      mol_list.append(Chem.MolFromSmiles(smiles))

  return mol_list

def make_initial_population(population_size,file_name):
  mol_list = read_file(file_name)
  population = []
  for i in range(population_size):
    population.append(random.choice(mol_list))
    
  return population


def calculate_normalized_fitness(population):
  fitness = []
  for gene in population:
    score = sc.logP_score(gene)
    fitness.append(max(float(score),0.0))
  
  #calculate probability
  sum_fitness = sum(fitness)
  normalized_fitness = [score/sum_fitness for score in fitness]

    
  return normalized_fitness

def make_mating_pool(population,fitness,mating_pool_size):
  mating_pool = []
  for i in range(mating_pool_size):
  	mating_pool.append(np.random.choice(population, p=fitness))

  return mating_pool
 

def reproduce(mating_pool,population_size,mutation_rate):
  new_population = []
  for n in range(population_size):
    parent_A = random.choice(mating_pool)
    parent_B = random.choice(mating_pool)
    #print Chem.MolToSmiles(parent_A),Chem.MolToSmiles(parent_B)
    new_child = co.crossover(parent_A,parent_B)
    #print new_child
    if new_child != None:
	    new_child = mu.mutate(new_child,mutation_rate)
	    #print "after mutation",new_child
	    if new_child != None:
	    	new_population.append(new_child)

  
  return new_population


if __name__ == "__main__":
    main()