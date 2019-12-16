from rdkit import Chem
import numpy as np
import time
import crossover as co
import scoring_functions as sc
import GB_GA as ga 
import sys
from multiprocessing import Pool
import pickle
import random

from rdkit import rdBase

n_tries = 1 #10
population_size = 20
mating_pool_size = 20
generations = 50
mutation_rate = 0.05
co.average_size = 39.15
co.size_stdev = 3.50
scoring_function = sc.logP_max
max_score = 9999.
scoring_args = []
n_cpus = 1
seeds = [0]

file_name = sys.argv[1]

print('* RDKit version', rdBase.rdkitVersion)
print('* population_size', population_size)
print('* mating_pool_size', mating_pool_size)
print('* generations', generations)
print('* mutation_rate', mutation_rate)
print('* max_score', max_score)
print('* average_size/size_stdev', co.average_size, co.size_stdev)
print('* initial pool', file_name)
print('* number of tries', n_tries)
print('* number of CPUs', n_cpus)
print('* seeds', ','.join(list(map(str, seeds))))
print('* ')
#remember to uncomment print statement in GB_GA reproduce
print('mut_child,new_child,parent_A,parent_B')

high_scores_list = []
count = 0
for prune_population in [True]:
    index = slice(0,n_tries) if prune_population else slice(n_tries,2*n_tries)
    temp_args = [[population_size, file_name,scoring_function,generations,mating_pool_size,
                  mutation_rate,scoring_args, max_score, prune_population] for i in range(n_tries)]
    args = []
    for x,y in zip(temp_args,seeds[index]):
        x.append(y)
        args.append(x)
    with Pool(n_cpus) as pool:
        output = pool.map(ga.GA, args)

    for i in range(n_tries):   
        #(scores, population) = ga.GA([population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate,scoring_args],prune_population)
        (scores, population, high_scores, generation) = output[i]
        smiles = Chem.MolToSmiles(population[0], isomericSmiles=True)
        high_scores_list.append(high_scores)
        #print(f'{i},{scores[0]:.2f},{smiles},{generation},Graph,{prune_population}')

pickle.dump(high_scores_list, open('test_try.p', 'wb' ))
