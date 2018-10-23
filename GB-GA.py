from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

import numpy as np
import random
import time
import sys

import sascorer
import crossover as co
import mutate as mu

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

def logP_score(m):
  try:
  	logp = Descriptors.MolLogP(m)
  except:
    print (m, Chem.MolToSmiles(m))
    sys.exit('failed to make a molecule')

  SA_score = -sascorer.calculateScore(m)
  #cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(m)))
  cycle_list = m.GetRingInfo().AtomRings() #remove networkx dependence
  if len(cycle_list) == 0:
      cycle_length = 0
  else:
      cycle_length = max([ len(j) for j in cycle_list ])
  if cycle_length <= 6:
      cycle_length = 0
  else:
      cycle_length = cycle_length - 6
  cycle_score = -cycle_length
  #print cycle_score
  #print SA_score
  #print logp
  SA_score_norm=(SA_score-SA_mean)/SA_std
  logp_norm=(logp-logP_mean)/logP_std
  cycle_score_norm=(cycle_score-cycle_mean)/cycle_std
  score_one = SA_score_norm + logp_norm + cycle_score_norm
  
  global max_score
  global count
  
  count += 1
  if score_one > max_score[0]:
    max_score = [score_one, Chem.MolToSmiles(m)]
  
  
  return score_one

def calculate_normalized_fitness(population):
  fitness = []
  for gene in population:
    score = logP_score(gene)
    fitness.append(max(float(score),0.0))
  
  #calculate probability
  sum_fitness = sum(fitness)
  normalized_fitness = [score/sum_fitness for score in fitness]

    
  return normalized_fitness

def make_mating_pool(population,fitness):
  mating_pool = []
  for i in range(population_size):
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


global max_score
global count

logP_values = np.loadtxt('logP_values.txt')
SA_scores = np.loadtxt('SA_scores.txt')
cycle_scores = np.loadtxt('cycle_scores.txt')
SA_mean =  np.mean(SA_scores)
SA_std=np.std(SA_scores)
logP_mean = np.mean(logP_values)
logP_std= np.std(logP_values)
cycle_mean = np.mean(cycle_scores)
cycle_std=np.std(cycle_scores)

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

file_name = '1000.smi'

results = []
size = []
t0 = time.time()
for i in range(10):
	max_score = [-99999.,'']
	count = 0
	population = make_initial_population(population_size,file_name)

	for generation in range(generations):
	  #if generation%10 == 0: print generation
	  fitness = calculate_normalized_fitness(population)
	  mating_pool = make_mating_pool(population,fitness)
	  population = reproduce(mating_pool,population_size,mutation_rate)

	print(i, max_score[0], max_score[1], Chem.MolFromSmiles(max_score[1]).GetNumAtoms())
	results.append(max_score[0])
	size.append(Chem.MolFromSmiles(max_score[1]).GetNumAtoms())

t1 = time.time()
print('')
print('time ',t1-t0)
print(max(results),np.array(results).mean(),np.array(results).std())
print(max(size),np.array(size).mean(),np.array(size).std())