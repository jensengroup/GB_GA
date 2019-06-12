'''
Written by Jan H. Jensen 2018
'''

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

import numpy as np
import sys
from multiprocessing import Pool

import sascorer

logP_values = np.loadtxt('logP_values.txt')
SA_scores = np.loadtxt('SA_scores.txt')
cycle_scores = np.loadtxt('cycle_scores.txt')
SA_mean =  np.mean(SA_scores)
SA_std=np.std(SA_scores)
logP_mean = np.mean(logP_values)
logP_std= np.std(logP_values)
cycle_mean = np.mean(cycle_scores)
cycle_std=np.std(cycle_scores)

def calculate_score(args):
  '''Parallelize at the score level (not currently in use)'''
  gene, function, scoring_args = args
  score = function(gene,scoring_args)
  return score

def calculate_scores_parallel(population,function,scoring_args, n_cpus):
  '''Parallelize at the score level (not currently in use)'''
  args_list = []
  args = [function, scoring_args]
  for gene in population:
    args_list.append([gene]+args)

  with Pool(n_cpus) as pool:
    scores = pool.map(calculate_score, args_list)

  return scores

def calculate_scores(population,function,scoring_args):
  scores = []
  for gene in population:
    score = function(gene,scoring_args)
    scores.append(score)

  return scores 

def logP_max(m, dummy):
  score = logP_score(m)
  return max(0.0, score)

def logP_target(m,args):
  target, sigma = args
  score = logP_score(m)
  score = GaussianModifier(score, target, sigma)
  return score


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
  
  return score_one

# GuacaMol article https://arxiv.org/abs/1811.09621
# adapted from https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/fingerprints.py

def get_ECFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2)

def get_ECFP6(mol):
    return AllChem.GetMorganFingerprint(mol, 3)

def get_FCFP4(mol):
    return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

def get_FCFP6(mol):
    return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)

def rediscovery(mol,args):
  target = args[0]
  try:
    fp_mol = get_ECFP4(mol)
    fp_target = get_ECFP4(target)

    score = TanimotoSimilarity(fp_mol, fp_target)

    return score
  
  except:
    print('Failed ',Chem.MolToSmiles(mol))
    return None

def similarity(mol,target,threshold):
  score = rediscovery(mol,target)
  if score:
    return ThresholdedLinearModifier(score,threshold)
  else:
    return None

# adapted from https://github.com/BenevolentAI/guacamol/blob/master/guacamol/score_modifier.py

def ThresholdedLinearModifier(score,threshold):
  return min(score,threshold)/threshold

def GaussianModifier(score, target, sigma):
  return np.exp(-0.5 * np.power((score - target) / sigma, 2.))




if __name__ == "__main__":
    pass
