import numpy as np

import scipy.stats as stats
import networkx as nx
import os 
from utils import HypergraphStructure as HS
from utils.HypergraphStructure import save_hypergraph_to_file

def Generate_PL_Hypergraph(gamma: float, kmin: int, N: int, m: int, save_file=None, verbose = False, delta_n = 10, max_iterations=10000) -> np.array:
    """
    Generate a hypergraph with a truncated power-law cardinality sequence.

    Parameters:
    - gamma (float): Exponent parameter governing the shape of the power-law distribution.
    - kmin (int): Minimum value for the generated cardinality sequence.
    - N (int): Desired number of nodes in the hypergraph.
    - m (int): Desired number of hyperedges in the hypergraph (may fluctuate).
    - save_file (str or None): Filename to save the hypergraph into a save_file.hyperedgelist file. 
                               If None, no file will be saved.
    - verbose (bool): Whether to print information about the generation process.
    - delta_n (int): how much to increase or decrease the number of nodes after 200 experiments
                     if convergence is not reach (default 10). This parameters allows faster convergence
                     but may change the number of hyperedges. If one wants to ensure that the number of 
                     hyperedges is strictly equal to m, then set delta_n = 0. 
    - max_iterations (int): Maximum number of experiments to avoid looping forever (default 1000).

    Returns:
    - np.array: A hypergraph represented as a 2D NumPy array, where each row corresponds
                to a hyperedge, and entries within a row represent the cardinality of 
                the hyperedge (first entry) and the node indices belonging to the hyperedge. 
    """

    n = N - delta_n
    gn = 0
    iterations = 0
    
    # Loop until the desired number of nodes (N) is reached in the largest connected component of the hypergraph
    while gn != N and iterations < max_iterations:
        
        # Adjust the number of nodes for faster convergence, may introduce fluctuations in the number of hyperedges
        if gn < N:
            n = n + delta_n
        else:
            n = n - delta_n

        # Calculate the maximum expected cardinality based on the square root of the number of nodes    
        emax = int(np.ceil(np.sqrt(n)))
        
        # Loop through 200 experiments to generate hypergraphs
        for exp in range(200):
            # Generate a truncated power-law cardinality sequence
            power_law_generator = truncated_power_law(gamma, emax, kmin)
            cardinalities = power_law_generator.rvs(size = m)
            
            hyper = generate_hyperedges(cardinalities, n)
            
            # Initialize an adjacency matrix for the hypergraph
            A = HS.Adjacency_HE_Normalized(hyper, n)
            
            # Create a NetworkX graph from the adjacency matrix
            G=nx.from_numpy_array(A)
            
            # Identify connected components in the hypergraph
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            gn = len(Gcc[0])
            
            # Print information about the experiment
            if verbose:
                print([gamma, exp, gn, n])
            
            # If the desired number of nodes is reached, process and return the hypergraph
            if gn == N:
                hyper = process_hypergraph(hyper, Gcc[0])

                # Save hypergraph into a specified file or do not save if save_file is None
                if save_file is not None:
                    save_hypergraph_to_file(hyper, N, save_file)
                return hyper
    # if convergence is not reached   
    raise RuntimeError("Maximum number of iterations reached without achieving the desired number of nodes.")

def Generate_Poisson_Hypergraph(beta: float, N: int, m: int, save_file=None, verbose = False, delta_n = 10, max_iterations=10000):
    """
    Generate a hypergraph with a Poisson cardinality distribution.

    Parameters:
    - beta (float): Poisson distribution parameter.
    - N (int): Desired number of nodes in the hypergraph.
    - m (int): Desired number of hyperedges in the hypergraph (may fluctuate).
    - save_file (str or None): Filename to save the hypergraph into a save_file.hyperedgelist file. 
                               If None, no file will be saved.
    - verbose (bool): Whether to print information about the generation process.
    - delta_n (int): how much to increase or decrease the number of nodes after 200 experiments
                     if convergence is not reach (default 10). This parameters allows faster convergence
                     but may change the number of hyperedges. If one wants to ensure that the number of 
                     hyperedges is strictly equal to m, then set delta_n = 0. 
    - max_iterations (int): Maximum number of experiments to avoid looping forever (default 1000).

    Returns:
    - np.array: A hypergraph represented as a 2D NumPy array, where each row corresponds
                to a hyperedge, and entries within a row represent the cardinality of 
                the hyperedge (first entry) and the node indices belonging to the hyperedge. 
    """

    n = N - delta_n
    gn = 0
    iterations = 0
    
    # Loop until the desired number of nodes (N) is reached in the largest connected component of the hypergraph
    while gn != N and iterations < max_iterations:
        
        # Adjust the number of nodes for faster convergence, may introduce fluctuations in the number of hyperedges
        if gn < N:
            n = n + delta_n
        else:
            n = n - delta_n
        
        # Loop through 200 experiments to generate hypergraphs
        for exp in range(200):
            # Generate a Poisson cardinality sequence
            cardinalities = generate_poisson_sequence(beta, m)

            # Generate hyperedges based on the sampled cardinalities
            hyper = generate_hyperedges(cardinalities, n)
            
            A = HS.Adjacency_HE_Normalized(hyper, n)
            
            G=nx.from_numpy_array(A)
            
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True);
            gn = len(Gcc[0])

            # Print information about the experiment
            if verbose:
                print([beta, exp, gn, n])
            
            # If the desired number of nodes is reached, process and return the hypergraph
            if gn == N:
                hyper = process_hypergraph(hyper, Gcc[0])

                # Save hypergraph into a specified file or do not save if save_file is None
                if save_file is not None:
                    save_hypergraph_to_file(hyper, N, save_file)
                return hyper
    # if convergence is not reached   
    raise RuntimeError("Maximum number of iterations reached without achieving the desired number of nodes.")
    
def Generate_Uniform_Hypergraph(cardinality: int, N: int, m: int, save_file=None, verbose = False, delta_n = 10, max_iterations=10000):
    """
    Generate a Uniform hypergraph with all hyperedges of size = cardinality.

    Parameters:
    - cardinality (int): Poisson distribution parameter.
    - N (int): Desired number of nodes in the hypergraph.
    - m (int): Desired number of hyperedges in the hypergraph (may fluctuate).
    - save_file (str or None): Filename to save the hypergraph into a save_file.hyperedgelist file. 
                               If None, no file will be saved.
    - verbose (bool): Whether to print information about the generation process (default False).
    - delta_n (int): how much to increase or decrease the number of nodes after 200 experiments
                     if convergence is not reach (default 10). This parameters allows faster convergence
                     but may change the number of hyperedges. If one wants to ensure that the number of 
                     hyperedges is strictly equal to m, then set delta_n = 0. 
    - max_iterations (int): Maximum number of experiments to avoid looping forever (default 1000).

    Returns:
    - np.array: A hypergraph represented as a 2D NumPy array, where each row corresponds
                to a hyperedge, and entries within a row represent the cardinality of 
                the hyperedge (first entry) and the node indices belonging to the hyperedge. 
    """

    # Initialize the number of nodes in the hypergraph with a buffer of delta_n
    n = N - delta_n
    gn = 0
    iterations = 0
    
    # Loop until the desired number of nodes (N) is reached in the largest connected component of the hypergraph
    while gn != N and iterations < max_iterations:
        
        # Adjust the number of nodes for faster convergence, may introduce fluctuations in the number of hyperedges
        if gn < N:
            n = n + delta_n
        else:
            n = n - delta_n

        # Loop through 200 experiments to generate hypergraphs
        for exp in range(200):
            cardinalities = [cardinality]*m

            # Generate hyperedges based on the sampled cardinalities
            hyper = generate_hyperedges(cardinalities, n)
            
            A = HS.Adjacency_HE_Normalized(hyper, n)
            
            G=nx.from_numpy_array(A)
            
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True);
            gn = len(Gcc[0])

            # Print information about the experiment
            if verbose:
                print([exp, gn, n])
            
            # If the desired number of nodes is reached, process and return the hypergraph
            if gn == N:
                hyper = process_hypergraph(hyper, Gcc[0])
                # Save hypergraph into a specified file or do not save if save_file is None
                if save_file is not None:
                    save_hypergraph_to_file(hyper, N, save_file)

                return hyper
        iterations +=200
    # if convergence is not reached   
    raise RuntimeError("Maximum number of iterations reached without achieving the desired number of nodes.")


def Generate_Uniform_PL_Hypergraph(cardinality, N, m, gamma, save_file=None, verbose = False, delta_n = 10, max_iterations=10000):
    """
    Generate a hypergraph with a uniform power-law cardinality sequence.

    Parameters:
    - cardinality (int): Cardinality of each hyperedge.
    - N (int): Desired number of nodes in the hypergraph.
    - m (int): Number of hyperedges to generate.
    - gamma (float): Exponent parameter governing the shape of the power-law distribution.
    - save_file (str or None): Filename to save the hypergraph into a save_file.hyperedgelist file. 
                               If None, no file will be saved.
    - verbose (bool): Whether to print information about the generation process.
    - delta_n (int): how much to increase or decrease the number of nodes after 200 experiments
                     if convergence is not reach (default 10). This parameters allows faster convergence
                     but may change the number of hyperedges. If one wants to ensure that the number of 
                     hyperedges is strictly equal to m, then set delta_n = 0. 
    - max_iterations (int): Maximum number of experiments to avoid looping forever (default 1000).

    Returns:
    - np.array: A hypergraph represented as a 2D NumPy array, where each row corresponds
                to a hyperedge, and entries within a row represent the node indices belonging to the hyperedge. 
    """

    n = N - delta_n
    gn = 0
    iterations = 0
    
    # Loop until the desired number of nodes (N) is reached in the largest connected component of the hypergraph
    while gn != N and iterations < max_iterations:
        
        # Adjust the number of nodes for faster convergence, may introduce fluctuations in the number of hyperedges
        if gn < N:
            n = n + delta_n
        else:
            n = n - delta_n
            
        p = node_probability_distribution(gamma, n)

        # Loop through 200 experiments to generate hypergraphs
        for exp in range(200):
        
            hyper = list()
            for i in range(m):
                # Select nodes based on the probability distribution
                nodes = np.random.choice(n,size=cardinality,replace = False, p=p)
                hyper.append(np.insert(nodes, 0, cardinality))
            
            hyper = np.array(hyper)
            
            A = HS.Adjacency_HE_Normalized(hyper, n)
            
            G=nx.from_numpy_array(A)
            
            Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
            gn = len(Gcc[0])

            if verbose:
                print([exp, gn, n])
            
            if gn == N:
                hyper = process_hypergraph(hyper, Gcc[0])

                # Save hypergraph into a specified file or do not save if save_file is None
                if save_file is not None:
                    save_hypergraph_to_file(hyper, N, save_file)

                return np.array(hyper)
    # if convergence is not reached   
    raise RuntimeError("Maximum number of iterations reached without achieving the desired number of nodes.")

def Generate_PL_PL_Hypergraph(N,M, gamma_nodes,gamma_edges,Runs, name = 'std',Randomize_swap = 10000):
    kmax = int(np.floor(np.sqrt(N)))
    emax = int(np.floor(np.sqrt(N)))
    if gamma_nodes >= 3.0:
        kmax = int(np.floor(2*N**(0.75/(gamma_nodes - 1))))
    if gamma_edges >= 3.0:
        emax = int(np.floor(2*N**(0.75/(gamma_edges - 1))))
    if name=='std':
        save_as = f'H_nodes_{N}_PL_{gamma_nodes:.3}_PL_{gamma_edges:.3}'
    else: 
        save_as=name
    os.system(f'./hypergraphstructure -GenerateRandomHypergraph n {N} Runs {Runs} Degree PL 2 {kmax} {gamma_nodes} '\
              f'Cardinality PL {2} {emax} {gamma_edges} -Randomize {Randomize_swap} -SaveHypergraph {save_as} -verbose')
    return

################## utility functions ##################

def generate_hyperedges(cardinalities, n):
    hyper = []
    for cardinality in cardinalities:
        nodes = np.random.choice(n, cardinality, replace=False)
        hyper.append(np.insert(nodes, 0, cardinality))
    return np.array(hyper, dtype=object)

def process_hypergraph(hyper, connected_component):
    k0 = list(connected_component)
    index2newIndex = {k0[i]: i for i in range(len(k0))}
    Del = []

    # Update hypergraph node indices based on the connected component
    for i in range(hyper.shape[0]):
        for j in range(1, hyper[i].shape[0]):
            if hyper[i][j] not in k0:
                Del.append(i)
            else:
                hyper[i][j] = index2newIndex[hyper[i][j]]

    # Remove rows corresponding to nodes not in the connected component
    Del = np.unique(np.array(Del))
    if Del.shape[0] > 0:
        hyper = np.delete(hyper, Del, 0)

    return hyper

def generate_poisson_sequence(beta, n):
    # Generate a Poisson sequence
    k = np.random.poisson(beta, size=2 * n)
    k = np.array(np.ceil(k), dtype=int)
    k = k[np.argwhere(k > 1)]
    k = k[0:n].reshape(n) #idk if reshape is necessary
    return k

def node_probability_distribution(gamma, n):
    # Calculate the probability distribution for node selection
    
    mu = 1/(gamma - 1); 
    p = np.array([(i + 1) ** (-mu) for i in range(n)])
    p /= np.sum(p)
    pc = np.cumsum(p)
    return p

def truncated_power_law(gamma: float, kmax: int, kmin: int)->stats.rv_discrete:
    """
    Generate a discrete random variable following a truncated power-law distribution.

    Parameters:
    - gamma (float): Exponent parameter governing the shape of the power-law distribution.
    - kmax (int): Maximum value for the generated discrete random variable.
    - kmin (int): Minimum value for the generated discrete random variable.

    Returns:
    - A scipy rv_descrete class used to generate discrete random variables following a truncated power-law distribution
                  over the range [kmin, kmax].
    """

    x = np.arange(kmin, kmax + 1, dtype='float')
    pmf = 1 / x**gamma
    pmf /= pmf.sum()

    # Create and return a random variable following the specified distribution
    return stats.rv_discrete(values=(range(kmin, kmax + 1), pmf))

 
def connected_component_subgraphs(G):
    """
    Generate connected component subgraphs of an undirected graph.

    Parameters:
    - G (networkx.Graph): The input undirected graph.

    Yields:
    - networkx.Graph: Connected component subgraphs of the input graph.
    
    This function uses NetworkX to identify connected components in the input graph (G)
    and yields each connected component as a subgraph.
    """
    for c in nx.connected_components(G):
        yield G.subgraph(c)
 

