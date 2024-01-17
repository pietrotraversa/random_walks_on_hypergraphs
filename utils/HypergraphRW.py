import numpy as np
from numpy import linalg as LA
from typing import Literal, Optional
from utils import HypergraphStructure as HS
import os

# RW coverage time - Standard random walk
# These functions take as argument the Laplacian matrix!

def Average_T_urw(Q: np.array) -> float:
    """
    Compute the global mean hitting time for an unbiased random walk on a graph.

    Parameters:
    - Q (np.array): The Laplacian matrix of the graph.

    Returns:
    - float: The global mean hitting time for the unbiased random walk.
    """
    
    Q = np.array(Q, dtype=np.double)
    
    # Calculate the sum of the diagonal elements of the Laplacian matrix (E)
    E = np.sum(np.diag(Q))
    
    # Get the number of nodes in the graph
    N = Q.shape[0]
    
    # Compute the eigenvalues and eigenvectors of the Laplacian matrix
    w, v = LA.eigh(Q)
    
    # Sort the eigenvalues in ascending order
    w = np.sort(w)
    
    # Exclude the smallest eigenvalue (0) and consider the rest
    w = w[range(1, N)]
    
    # Compute the global mean hitting time using the eigenvalues
    mean_hitting_time = E / (N - 1) * np.sum(1 / w)
    
    return mean_hitting_time


### partial mean hitting time for the unbiased random walk
def Tj_urw(Q: np.array) -> np.array:
    """
    Calculate the partial mean hitting time for the unbiased random walk.

    Parameters:
    - Q (np.array): Laplacian matrix of the graph.

    Returns:
    - np.array: Partial mean hitting time for each node.
    """
    Q = np.array(Q,dtype=np.double)
    E = np.sum(np.diag(Q))
    N = Q.shape[0]
    
    w, V = LA.eigh(Q)
    idx = w.argsort()
    w = w[idx]
    V = V[:,idx]

    
    v = V[:,range(1,N)]
    w = w[range(1,N)]
    
    k = np.diag(Q)

    T = np.zeros(N)
    
    for j in range(0,N):
        v_tmp = np.zeros(N-1)
        for i in range(1,N):
            v_tmp[i-1] = np.dot(k, V[:,i])
        
        tmp = (E*v[j,:]**2 - v[j,:]*v_tmp)
        T[j] = N/(N-1)*np.matmul((1/w.transpose()),tmp)


    return T

def Tij_urw(Q):
    """
    Compute the expected hitting time matrix for an Unbiased Random Walk (URW) on a graph Laplacian matrix.

    Parameters:
    - Q (numpy.array): Graph Laplacian matrix.

    Returns:
    - numpy.array: Matrix T where T[i, j] represents the expected time to hit node j starting from node i.
    """
    Q = np.array(Q,dtype=np.double)
    N = Q.shape[0];
    
    w, v = LA.eigh(Q)
    idx = w.argsort()
    w = w[idx]
    v = v[:,idx]
    
    d = np.diag(Q)
    
    T = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                tmp = np.sum((v[i, 1:] * v[:, 1:] - v[i, 1:] * v[j, 1:] - v[j, 1:] * v[:, 1:] + v[j, 1:]**2) / w[1:], axis=1)
                T[i, j] +=np.sum(d*tmp)
                # equivalent to
                #for z in range(N):
                #    tmp = 0;
                #    for k in range(1,N):
                #        tmp = tmp + 1/w[k]*(v[i,k]*v[z,k] - v[i,k]*v[j,k] - v[j,k]*v[z,k] + v[j,k]**2)   
                #    T[i,j] = T[i,j] + d[z]*tmp
                
    return T

# RW coverage time - Maximum entropy random walk
# This function takes as argument the Adjacency matrix!

### partial mean hitting time for the maximal entropy random walk
def Tj_merw(A: np.array) -> np.array:
    """
    Calculate the partial mean hitting time for the maximal entropy random walk.

    Parameters:
    - A (np.array): Adjacency matrix of the graph.

    Returns:
    - np.array: Partial mean hitting time for each node.
    """
    A = np.array(A,dtype=np.double)
    N = A.shape[0]
    
    w, v = LA.eigh(A)
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]

    T = np.zeros(N)
    
    for j in range(0,N):
        
        v_tmp = np.zeros(N-1)
        for k in range(1,N):
            v_tmp[k-1] = np.sum(v[:,k]/v[:,0])

        tmp1 = (N*v[j,range(1,N)]**2 - v[j,range(1,N)]*v[j,0]*v_tmp)
        tmp2 = w[0]/(w[0] - w[range(1,N)]);

        T[j] = 1/(v[j,0]**2*(N-1))*np.sum(tmp1*tmp2)
        
    return T

### expected time for a maximal entropy random walk to hit node v_j starting from v_i
def Tij_merw(A: np.array) -> np.array:
    """
    Calculate the expected time for a maximal entropy random walk to hit node v_j starting from v_i.

    Parameters:
    - A (np.array): Adjacency matrix of the graph.

    Returns:
    - np.array: Pairwise mean hitting time matrix.
    """
    A = np.array(A,dtype=np.float64);
    N = A.shape[0];
    
    w, v = LA.eigh(A);
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]

    T = np.zeros([N,N])
    
    for i in range(0,N):
        for j in range(0,N):
            if i != j:
                tmp = 0
                for k in range(1,N):
                    tmp1 = v[j,k]**2 - v[i,k]*v[j,k]*v[j,0]/v[i,0]
                    tmp2 = w[0]/(w[0] - w[k])
                    tmp = tmp + tmp2*tmp1
                       
                T[i,j] = 1/(v[j,0]**2)*tmp
                                    
    return T


### simulating randomw walks

def rw_on_generated_hypergraphs(generation_function, kwargs: dict, Nexp: int,
                                rw_type: Literal['urw','merw'] = 'urw',
                                rw_step: Literal['p','projected','ho','higher-order'] = 'ho',
                                save_to: Optional[str] = None):
    """
    Perform random walks on hypergraphs generated using a specified generation function.

    Parameters:
    - generation_function (callable): the function used to generate the hypergraphs.
    - kwargs (dict): Dictionary of parameters required for the generation function.
    - Nexp (int): Number of experiments or iterations.
    - rw_type (Literal['urw','merw']): Type of random walk. Options: 'urw' (unbiased random walk) or 'merw' (maximal entropy random walk). Default is 'urw'.
    - rw_step (Literal['p','projected','ho','higher-order']): Type of random walk step. Options: 'p' or 'projected' for the projected step, 'ho' or 'higher-order' for the higher-order step. Deafult is 'ho'.
    - save_to (Optional[str]): File path to save the hitting times data. If None, data is not saved.

    Returns:
    - np.array: Array containing mean and standard deviation of hitting times across experiments.
    """

    T = np.zeros(Nexp)
    i0 = 0
    N = kwargs['N']

    # Load existing data if available
    if save_to!=None and os.path.isfile(save_to) == True:
        Data = np.loadtxt(save_to, delimiter='\t')
        T[range(Data.shape[0])] = Data
        i0 = np.min(np.append(np.argwhere(T == 0), Nexp))
        print('file exists... starting from experiment',i0)

    # Perform experiments
    for i in range(i0, Nexp):
        H = generation_function(**kwargs)
        
        # Choose random walk step type
        if rw_step == 'ho' or rw_step == 'higher-order':
            A = HS.Adjacency_HE_Normalized(H, N) 
        elif rw_step == 'p' or rw_step == 'projected':
            A = HS.Adjacency_Count(H,N)
        else:
            raise ValueError(f'{rw_step} not a valid type of random walk step. Valid inputs are "p" or "projected" for the projected step, or "ho" or "higher-order" for the higher-order step')
        
        # Choose random walk type
        if rw_type == 'urw':      
            L = HS.Laplacian(A)
            T[i] = Average_T_urw(L)
        elif rw_type == 'merw':
            T[i] = np.mean(Tj_merw(A))
        else:
            raise ValueError(f'{rw_type} not a valid type of random walk. Valid inputs are "urw" for unbiased random walk or "merw" for maximal entropy random walk')
        
        # Save hitting times data
        if save_to is not None:
            np.savetxt(save_to, T, delimiter='\t')
    
    return np.array([np.mean(T), np.std(T)])
