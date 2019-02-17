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

def calculate_scores(population):
  scores = []
  for gene in population:
    score = sc.logP_score(gene)
    scores.append(max(float(score),0.0))

  return scores 

def calculate_normalized_fitness(scores):  
  #calculate probability
  sum_scores = sum(scores)
  normalized_fitness = [score/sum_scores for score in scores]

  return normalized_fitness



if __name__ == "__main__":
  global max_score
  global count


  population_size = 20 
  mating_pool_size = 20
  generations = 50
  mutation_rate = 0.01

  co.average_size = 39.15
  co.size_stdev = 3.50

  print('population_size', population_size)
  print('mating_pool_size', mating_pool_size)
  print('generations', generations)
  print('mutation_rate', mutation_rate)
  print('average_size/size_stdev', co.average_size, co.size_stdev)
  print('')

  file_name = 'ZINC_first_1000.smi'

  results = []
  size = []
  t0 = time.time()
  all_scores = []
  for i in range(5):
      sc.max_score = [-99999.,'']
      sc.count = 0
      population = ga.make_initial_population(population_size,file_name)
      scores = calculate_scores(population)
      fitness = calculate_normalized_fitness(scores)

      for generation in range(generations):
        mating_pool = ga.make_mating_pool(population,fitness,mating_pool_size)
        new_population = ga.reproduce(mating_pool,population_size,mutation_rate)
        new_scores = calculate_scores(new_population)
        population_tuples = list(zip(scores+new_scores,population+new_population))
        population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
        population = [t[1] for t in population_tuples]
        scores = [t[0] for t in population_tuples]  
        fitness = calculate_normalized_fitness(scores)

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