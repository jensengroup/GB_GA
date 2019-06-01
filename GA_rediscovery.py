from rdkit import Chem
import numpy as np
import time
import crossover as co
import scoring_functions as sc
import GB_GA as ga 

Celecoxib = 'O=S(=O)(c3ccc(n1nc(cc1c2ccc(cc2)C)C(F)(F)F)cc3)N'
target = Chem.MolFromSmiles(Celecoxib)
n_tries = 1
population_size = 20 
mating_pool_size = 20
generations = 20
mutation_rate = 0.5
co.average_size = target.GetNumAtoms() 
co.size_stdev = 5
scoring_function = sc.rediscovery
scoring_args = [target]
n_cpus = 2

file_name = 'ZINC_first_1000.smi'
file_name = 'Celecoxib_1000_50.smi'
file_name = 'Celecoxib_1000_100.smi'

print('target', Celecoxib)
print('population_size', population_size)
print('mating_pool_size', mating_pool_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('average_size/size_stdev', co.average_size, co.size_stdev)
print('initial pool', file_name)
print('number of tries', n_tries)
print('number of CPUs', n_cpus)
print('')

results = []
size = []
t0 = time.time()
all_scores = []
for i in range(n_tries):     
    scores, population = ga.GA(population_size, file_name,scoring_function,generations,mating_pool_size, 
                               mutation_rate,scoring_args,n_cpus)
    all_scores.append(scores)
    print(i, scores[0], Chem.MolToSmiles(population[0]))
    results.append(scores[0])
    #size.append(Chem.MolFromSmiles(sc.max_score[1]).GetNumAtoms())

t1 = time.time()
print('')
print('time ',t1-t0)
print(max(results),np.array(results).mean(),np.array(results).std())
#print(max(size),np.array(size).mean(),np.array(size).std())

#print(all_scores)