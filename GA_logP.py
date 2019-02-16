'''
Written by Jan H. Jensen 2018
'''
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

import numpy as np
import random
import time
import sys

import crossover as co
import mutate as mu
import scoring_functions as sc
import GB_GA as ga 

def calculate_normalized_fitness(population):
  fitness = []
  for gene in population:
    score = sc.logP_score(gene)
    fitness.append(max(float(score),0.0))

  #calculate probability
  sum_fitness = sum(fitness)
  normalized_fitness = [score/sum_fitness for score in fitness]

  return normalized_fitness

def calculate_normalized_fitness2(population):
  fitness = []
  for gene in population:
    score = sc.logP_score(gene)
    #print('score',score)
    fitness.append(float(score))

  #calculate probability
  new_fitness = [score - min(fitness) + 1. for score in fitness]
  #print('new_fitness',new_fitness)
  sum_fitness = sum(new_fitness)
  normalized_fitness = [score/sum_fitness for score in new_fitness]



if __name__ == "__main__":
  global max_score
  global count


  population_size = 20 
  generations = 50
  mutation_rate = 0.01

  co.average_size = 39.15
  co.size_stdev = 3.50

  print('population_size', population_size)
  print('generations', generations)
  print('mutation_rate', mutation_rate)
  print('average_size/size_stdev', co.average_size, co.size_stdev)
  print('')

  file_name = 'ZINC_first_1000.smi'

  results = []
  size = []
  t0 = time.time()
  all_scores = []
  for i in range(10):
    sc.max_score = [-99999.,'']
    sc.count = 0
    population = ga.make_initial_population(population_size,file_name)

    scores = []
    for generation in range(generations):
      #if generation%10 == 0: print generation
      fitness = calculate_normalized_fitness(population)
      mating_pool = ga.make_mating_pool(population,fitness,population_size)
      population = ga.reproduce(mating_pool,population_size,mutation_rate)
      scores.append(sc.max_score[0])

    all_scores.append(scores)
    print(i, sc.max_score[0], sc.max_score[1], Chem.MolFromSmiles(sc.max_score[1]).GetNumAtoms())
    results.append(sc.max_score[0])
    size.append(Chem.MolFromSmiles(sc.max_score[1]).GetNumAtoms())

  t1 = time.time()
  print('')
  print('time ',t1-t0)
  print(max(results),np.array(results).mean(),np.array(results).std())
  print(max(size),np.array(size).mean(),np.array(size).std())

  #print(all_scores)