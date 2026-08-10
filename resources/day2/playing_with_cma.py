# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:36:40 2025

@author: ozbejv
"""

import cma
import numpy as np
import time
from multiprocessing import Pool

def loss(x):
    l = 0
    time.sleep(0.01)
    for i in x:
        l+=i**2
    return l

if __name__ == "__main__":

    n_proc = 4
    es = cma.CMAEvolutionStrategy(5*[1.1], 0.1, {"bounds": [5*[-1], 5*[2]], "tolfun": 10**-9})
    pool = Pool(n_proc)
    while not es.stop():
        sols = es.ask()
        
        f_evals = pool.map(loss, sols)
        
        es.tell(sols, f_evals)
        es.disp()
        
    pool.close()
    pool.join()
    es.result_pretty()