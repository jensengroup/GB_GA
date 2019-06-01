# GB-GA
[Graph-based genetic algorithm](http://dx.doi.org/10.1039/C8SC05372C)
 
usage example: 'python GA_logP.py ZINC_first_1000.smi' or 'python GA_rediscovery.py Celecoxib_1000_100.smi'. The idea is that the py file serves as an input file.

The calculation of scores for a population is parallelized. For fast scoring functions such as logP and rediscovery (Tanimoto score) not much is gained by using more than 1 CPU as time is mostly spend on constructing the populations. 
