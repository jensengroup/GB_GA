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
import GB_GA as ga 

def calculate_normalized_fitness(population,target):
  fitness = []
  for gene in population:
    score = sc.rediscovery(gene,target)
    fitness.append(score)
  
  #calculate probability
  sum_fitness = sum(fitness)
  if sum_fitness < 0.01:
    print([Chem.MolToSmiles(gene) for gene in population])
  normalized_fitness = [score/sum_fitness for score in fitness]

  return normalized_fitness



if __name__ == "__main__":
  global max_score
  global count

  Celecoxib = 'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
  target = Chem.MolFromSmiles(Celecoxib)
  target = Chem.MolFromSmiles('CCCCCCCCCCCCCCCCCCCCCCCCC')

  population_size = 20 
  generations = 50
  mutation_rate = 0.01

  co.average_size = target.GetNumAtoms()
  co.size_stdev = 5

  print('population_size', population_size)
  print('generations', generations)
  print('mutation_rate', mutation_rate)
  print('average_size/size_stdev', co.average_size, co.size_stdev)
  print('')

  file_name = 'ZINC_first_1000.smi'

  results = []
  size = []
  t0 = time.time()
  for i in range(1):
    sc.max_score = [-99999.,'']
    sc.count = 0
    population = ga.make_initial_population(population_size,file_name)

    for generation in range(generations):
      fitness = calculate_normalized_fitness(population,target)
      mating_pool = ga.make_mating_pool(population,fitness,population_size)
      population = ga.reproduce(mating_pool,population_size,mutation_rate)
      if generation % 10 == 0:
        print(generation, sc.max_score[0], sc.max_score[1], Chem.MolFromSmiles(sc.max_score[1]).GetNumAtoms())

    print(i, sc.max_score[0], sc.max_score[1], Chem.MolFromSmiles(sc.max_score[1]).GetNumAtoms())
    results.append(sc.max_score[0])
    size.append(Chem.MolFromSmiles(sc.max_score[1]).GetNumAtoms())

  t1 = time.time()
  print('')
  print('time ',t1-t0)
  print(max(results),np.array(results).mean(),np.array(results).std())
  print(max(size),np.array(size).mean(),np.array(size).std())