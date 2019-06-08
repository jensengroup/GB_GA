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
#import string_crossover as co
#import string_mutate as mu
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

def calculate_normalized_fitness(scores):
  sum_scores = sum(scores)
  normalized_fitness = [score/sum_scores for score in scores]

  return normalized_fitness

def make_mating_pool(population,fitness,mating_pool_size):
  mating_pool = []
  for i in range(mating_pool_size):
  	mating_pool.append(np.random.choice(population, p=fitness))

  return mating_pool
 

def reproduce(mating_pool,population_size,mutation_rate):
  new_population = []
  while len(new_population) < population_size:
    parent_A = random.choice(mating_pool)
    parent_B = random.choice(mating_pool)
    new_child = co.crossover(parent_A,parent_B)
    if new_child != None:
	    new_child = mu.mutate(new_child,mutation_rate)
	    if new_child != None:
	    	new_population.append(new_child)

  return new_population

def GA(args):
  population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate, \
  scoring_args, max_score = args

  population = make_initial_population(population_size,file_name)
  scores = sc.calculate_scores(population,scoring_function,scoring_args)
  fitness = calculate_normalized_fitness(scores)

  for generation in range(generations):
    mating_pool = make_mating_pool(population,fitness,mating_pool_size)
    new_population = reproduce(mating_pool,population_size,mutation_rate)
    new_scores = sc.calculate_scores(new_population,scoring_function,scoring_args)
    population_tuples = list(zip(scores+new_scores,population+new_population))
    population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
    population = [t[1] for t in population_tuples]
    scores = [t[0] for t in population_tuples]  
    fitness = calculate_normalized_fitness(scores)
    if scores[0] >= max_score:
      break

  return (scores, population, generation+1)


if __name__ == "__main__":
    pass