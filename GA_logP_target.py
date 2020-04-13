from rdkit import Chem
import numpy as np
import time
import crossover as co
import scoring_functions as sc
import GB_GA as ga 
import sys
from multiprocessing import Pool

n_tries = 10
population_size = 20 
mating_pool_size = 20
generations = 100
mutation_rate = 0.01
co.average_size = 50.
co.size_stdev = 5
scoring_function = sc.logP_target
max_score = 0.99
target = -1.
sigma = 2.
scoring_args = [target,sigma]
prune_population = False
n_cpus = 8

file_name = sys.argv[1]

print('population_size', population_size)
print('mating_pool_size', mating_pool_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('max_score', max_score)
print('average_size/size_stdev', co.average_size, co.size_stdev)
print('initial pool', file_name)
print('prune_population', prune_population)
print('target +/ sigma', target, sigma)
print('number of tries', n_tries)
print('number of CPUs', n_cpus)
print('')

results = []
size = []
t0 = time.time()
all_scores = []
generations_list = []
args = n_tries*[[population_size, file_name,scoring_function,generations,mating_pool_size,
                 mutation_rate,scoring_args, max_score, prune_population]]
with Pool(n_cpus) as pool:
    output = pool.map(ga.GA, args)

for i in range(n_tries):     
    #(scores, population) = ga.GA([population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate,scoring_args],prune_population)
    (scores, population, generation) = output[i]
    all_scores.append(scores)
    print(f'{i} {scores[0]:.2f} {Chem.MolToSmiles(population[0])} {generation}')
    results.append(scores[0])
    generations_list.append(generation)
    #size.append(Chem.MolFromSmiles(sc.max_score[1]).GetNumAtoms())

t1 = time.time()
print('')
print(f'max score {max(results):.2f}, mean {np.array(results).mean():.2f} +/- {np.array(results).std():.2f}')
print(f'max generation {max(generations_list):.2f}, mean generations {np.array(generations_list).mean():.2f} +/- {np.array(generations_list).std():.2f}')
print(f'time {(t1-t0)/60.0:.2f} minutes')
#print(max(size),np.array(size).mean(),np.array(size).std())

#print(all_scores)