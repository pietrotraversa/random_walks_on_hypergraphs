import numpy as np
import networkx as nx
import numpy.linalg as LA
from typing import Literal

################ matrices ################

def Adjacency_HE_Normalized(hyper: np.array, n: int):
    """
    Compute the normalized adjacency matrix representing the connections between nodes in a hypergraph,
    considering hyperedge cardinalities.

    Parameters:
    - hyper (np.array): A 2D array representing a hypergraph. Each row corresponds to a hyperedge, 
                       and elements in each row represent the nodes connected by the hyperedge.
    - n (int): The total number of nodes in the hypergraph.

    Returns:
    - np.array: The normalized adjacency matrix. The element A[i][j] represents the weight of the connection 
                between nodes i and j, considering hyperedge cardinalities.
    """
    A = np.zeros((n, n))

    for hyperedge in hyper:
        cardinality = hyperedge[0]
        nodes = hyperedge[1:]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]
                weight = 1.0 / (cardinality - 1.0)
                A[node_i][node_j] += weight
                A[node_j][node_i] += weight

    return A


def Adjacency_Count(hyper: np.array, n: int):
    """
    Compute the counting adjacency matrix of the hypergraph.

    Parameters:
    - hyper (np.array): A 2D array representing a hypergraph. Each row corresponds to a hyperedge, 
                       and elements in each row represent the nodes connected by the hyperedge.
    - n (int): The total number of nodes in the hypergraph.

    Returns:
    - A (np.array): The counting adjacency matrix of the hypergraph.
                The element A[i][j] contains the count of hyperedges shared by nodes i and j.
    """
    A = np.zeros((n, n))

    for hyperedge in hyper:
        nodes = hyperedge[1:]  # Exclude the first element which represents the hyperedge size
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, node_j = nodes[i], nodes[j]
                A[node_i][node_j] += 1.0
                A[node_j][node_i] += 1.0

    return A


def Laplacian(A: np.array) -> np.array:
    """
    Compute the Laplacian matrix L = D - A of an input graph represented by its adjacency matrix.

    Parameters:
    - A (np.array): The adjacency matrix of the graph.

    Returns:
    - np.array: The Laplacian matrix of the input graph, computed as the difference between 
                the diagonal degree matrix and the adjacency matrix.
    """
    
    # Compute the diagonal degree matrix D as the row sums of the adjacency matrix
    D = np.diag(np.sum(A, 1))
    
    # Compute the Laplacian matrix as the difference between the diagonal degree matrix and the adjacency matrix
    L = D - A
    
    # Return the Laplacian matrix
    return L


def Probability_Transition(A: np.array, rw_type: Literal['URW','MERW'] = 'URW') -> np.array:
    """
    Compute the transition matrix of a random walk given its adjacency matrix.

    Parameters:
    - A (np.array): The adjacency matrix.
    - rw_type (Union['URW', 'MERW']): Type of random walk. 'URW' for unbiased random walk, 
                                     'MERW' for Maximal Entropy Random Walk (default 'URW').

    Returns:
    - np.array: The transition probability matrix of the random walk.
    """
    
    if rw_type=='URW':
        D_inv = np.diag(1 / np.sum(A, 1))
        P = np.matmul(D_inv, A)
        return P
    elif rw_type=='MERW':
        w, v = LA.eigh(A)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:,idx]
        N = A.shape[0]
        P = (A / w[0]) * (v[:, 0] / v[:, 0][:, np.newaxis])
        # equivalent to
        #P = np.zeros([N,N])
        #for i in range(N):
        #    for j in range(N):
        #        P[i,j] = A[i,j]/w[0]*v[j,0]/v[i,0]
        return P
    else:
        raise ValueError("Invalid random walk type. Use 'URW' for unbiased random walk or 'MERW' for maximal entropy random walk.")

def get_stationary_distribution(A: np.array, rw_type: Literal['URW','MERW'] = 'URW'):
    """
    Calculate the stationary distribution for a given adjacency matrix.

    Parameters:
    - A (np.array): Adjacency matrix.
    - rw_type (Union['URW', 'MERW']): Type of random walk. 'URW' for unbiased random walk, 
                                     'MERW' for Maximal Entropy Random Walk (default 'URW').

    Returns:
    - np.array: Stationary distribution vector.
    """
    if rw_type == 'URW':
        # Unbiased Random Walk
        pi = np.sum(A,axis=1)/np.sum(A)
        return pi
    elif rw_type == 'MERW':
        # Maximal Entropy Random Walk
        w, v = LA.eigh(A)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:,idx]
        phi = v[:,0]**2/(np.sum(v[:,0]**2))
        return phi
    else:
        raise ValueError("Invalid random walk type. Use 'URW' for unbiased random walk or 'MERW' for maximal entropy random walk.")

def Giant_Component_Hypergraph(hyper_in: np.array, n: int)->list:
    '''
    Extracts the giant connected component from a hypergraph represented as an edge list.

    Parameters:
    - hyper_in (np.array): Hypergraph edge list, where each row corresponds
                            to a hyperedge, and entries within a row represent the cardinality (first entry)
                            and the node indices belonging to the hyperedge.
    - n (int): Number of nodes in the hypergraph.

    Returns:
    - List: A list containing the following elements:
        - gn (int): Number of nodes in the giant connected component.
        - hyper (np.array): Hypergraph represented as a 2D NumPy array, where each row corresponds
                            to a hyperedge, and entries within a row represent the cardinality (first entry)
                            and the node indices belonging to the hyperedge.
    '''
    from utils.HypergraphModels import process_hypergraph
    
    A = Adjacency_Count(hyper_in, n)
    
    G=nx.from_numpy_array(A)
            
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    gn = len(Gcc[0])
    
    hyper = process_hypergraph(hyper_in, Gcc[0])

    return [gn, hyper];

################ read/write ################

def save_hypergraph_to_file(hyper,n, filename):
    # Write the hypergraph into a specified file
    with open(filename + '.hyperedgelist', 'w') as f:
        f.write(f"{n}\t{len(hyper)}\n")

        for e in hyper:
            for i in range(len(e)):
                f.write(f"{e[i]}")

                if i < len(e) - 1:
                    f.write("\t")

            f.write("\n")

def Read_simplex_Benson(data_name):
    '''
    Reads hypergraph data from Benson's collection of simplicial complexes.

    Parameters:
    - data_name (str): Name of the dataset, used to construct the filenames for vertex counts and simplices.

    Returns:
    - List: A list containing the following elements:
        - n (int): Number of nodes in the hypergraph.
        - hyper (np.array): Hypergraph represented as a 2D NumPy array, where each row corresponds
                            to a hyperedge, and entries within a row represent the cardinality (first entry)
                            and the node indices belonging to the hyperedge.
    '''
    hyper = []
    
    # load data
    file_name = "%s-nverts.txt" % (data_name)
    Cardinality = np.loadtxt(file_name, dtype=int)
    
    file_name = "%s-simplices.txt" % (data_name)
    Data_hedges = np.loadtxt(file_name, dtype=int)

    n = int(np.max(Data_hedges))

    Data_hedges = Data_hedges -1 # we start to count indices from zero

    i = 0

    # Process simplices data to construct hypergraph representation
    for cardinality in Cardinality:
        cardinality = int(cardinality)
        row = np.zeros(cardinality+1)
        row[0] =  cardinality
        row[range(1,cardinality+1)] = Data_hedges[range(i, i + cardinality)]
        i = i+cardinality
        
        hyper.append(np.array(row, dtype = int))
        
    hyper = np.array(hyper, dtype=object)
    
    return [n, hyper]


def Read_Hypergraph_data_Benson(data_name):
    '''
    Reads hypergraph data from a file in Benson's format.

    Parameters:
    - data_name (str): Name of the dataset, used to construct the filename.

    Returns:
    - List: A list containing the following elements:
        - n (int): Number of nodes in the hypergraph.
        - hyper (np.array): Hypergraph represented as a 2D NumPy array, where each row corresponds
                            to a hyperedge, and entries within a row represent the cardinality (first entry)
                            and the node indices belonging to the hyperedge.
    '''
    hyper = []
    n = 0
    
    file_name = "%s/hyperedges.txt" % (data_name)

    with open(file_name) as f:
        for line in f:
            row = [int(element)-1 for element in  line.split()]        
            hyper.append(np.insert(row,0,len(row))) #cardinality is the first element
            
            if n < np.max(row):
                n = np.max(row)
        
        
    hyper = np.array(hyper,dtype=object)
    
    return [n+1, hyper]
