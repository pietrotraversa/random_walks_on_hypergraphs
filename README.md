# URW and MERW on hypergraphs
Repository for the study "From unbiased to maximal entropy random walks on hypergraphs" (DOI: https://doi.org/10.1103/PhysRevE.109.054309).
***
## Hypergraph's convention
For this code, we represent a hypergraph as a hyperedgelist, i.e. a list of hyperedges. However, be careful that the first entry of each hyperedges is the cardinality of the hyperedges. This convention may be different from other hyperedgelist formats.
For example, the hypergraph with hyperedge set $\mathcal{E}$ = {[1,2],[2,3,4]} is represented as:
```python
H = [[2,1,2],
     [3,2,3,4]]
```
and the cardinalities are given by: 
```python
cardinalities = [e[0] for e in H]
```
## Utils
All the important functions are gathered inside this folder and are divided into 3 macro-categories:

* `HypergraphModels.py` contains all the functions we used in the paper to generate synthetic hypergraphs.
* `HypergraphStructure.py` contains a series of utility functions for hypergraph. For example, here one can find the code to calculate the adjacency or laplacian matrix, get the giant component, and read and write the hypergraph.
* `HypergraphRW.py` contains all the functions to compute hitting times and stationary distributions of random walk on hypergraphs.

## Notebook
This folder is a collection of examples to understand how the various functions in utils work. In this folder are also present the notebook used to plot the figure in the paper. Specifically:

* `example_experiments.ipynb`: example on how to perform numerical experiments on generated hypergraph.
* `example_HypergraphModel.ipynb`: example of how the module HypergraphModel can be used to generate hypergraphs with different degree and cardinality distributions.
* `Toy_hypergraph`: Toy example that offers a visual interpretation of how the different types of random walk. This notebook also reproduces the figures in the paper that are related to this example.
* `Real_data`: Mean hitting times of real hypergraphs.
* `synthetic_hypergraphs`: Results of the numerical experiments performed for the paper (requires running the script in the folder `python`).

## Python
Contains the script used to perform numerical experiments with different degree and cardinality distributions.

## Cite As
P Traversa, GF de Arruda, Y Moreno, Phys. Rev. E 109, 2024
