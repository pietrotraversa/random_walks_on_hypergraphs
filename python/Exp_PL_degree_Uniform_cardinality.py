import os
# change directory to the root of the repository. Execute only once!
# All utils assume that the working directory is the root directory of the github folder
os.chdir('../')
import sys
# Add utils directory in the list of directories to look for packages to import
sys.path.insert(0,os.getcwd())

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import HypergraphRW as HRW
from utils import HypergraphModels as HM

def update_dict(d:dict, key, value):
    # utility function for parallelization
    d.update({key:value})
    return d

num_cores = 10
gamma = np.unique(np.linspace(2.1, 3.5, 29))

function_arguments = {}
function_arguments['N'] = 1000
function_arguments['m'] = 1000
function_arguments['cardinality'] = 20
Nexp = 1000

# unbiased random walk with higher-order step
# check that the directories exist
save_to = 'data/pl_uniform/exp_urw_ho'
processed_list = Parallel(n_jobs=num_cores)(delayed(HRW.rw_on_generated_hypergraphs)(HM.Generate_Uniform_PL_Hypergraph,
                                                                                     update_dict(function_arguments,'gamma',i),
                                                                                     Nexp,
                                                                                     'urw',
                                                                                     'ho',
                                                                                     save_to + f'_gamma_{i:.2f}.txt') for i in tqdm(gamma))

# unbiased random walk with projected step
save_to = 'data/pl_uniform/exp_urw_p'
processed_list = Parallel(n_jobs=num_cores)(delayed(HRW.rw_on_generated_hypergraphs)(HM.Generate_Uniform_PL_Hypergraph,
                                                                                     update_dict(function_arguments,'gamma',i),
                                                                                     Nexp,
                                                                                     'urw',
                                                                                     'p',
                                                                                     save_to + f'_gamma_{i:.2f}.txt') for i in tqdm(gamma))

# maximal entropy random walk with higher-order step
save_to = 'data/pl_uniform/exp_merw_ho'
processed_list = Parallel(n_jobs=num_cores)(delayed(HRW.rw_on_generated_hypergraphs)(HM.Generate_Uniform_PL_Hypergraph,
                                                                                     update_dict(function_arguments,'gamma',i),
                                                                                     Nexp,
                                                                                     'merw',
                                                                                     'ho',
                                                                                     save_to + f'_gamma_{i:.2f}.txt') for i in tqdm(gamma))

# maximal entropy random walk with projected step
save_to = 'data/pl_uniform/exp_merw_p'
processed_list = Parallel(n_jobs=num_cores)(delayed(HRW.rw_on_generated_hypergraphs)(HM.Generate_Uniform_PL_Hypergraph,
                                                                                     update_dict(function_arguments,'gamma',i),
                                                                                     Nexp,
                                                                                     'merw',
                                                                                     'p',
                                                                                     save_to + f'_gamma_{i:.2f}.txt') for i in tqdm(gamma))