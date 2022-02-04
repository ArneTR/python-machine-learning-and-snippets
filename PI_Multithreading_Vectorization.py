#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:22:28 2020

@author: light
"""

import timeit
import multiprocessing as mp
import numpy as np
import random


def count_inside_point(n):
    m = 0
    for i in range(1,n):
        p_x = np.random.uniform(1, -1, 1)
        p_y = np.random.uniform(1, -1, 1)
        if (p_x**2 + p_y**2) <=1:
            m = m+1 
    return m


# now let's try the parallel approach
# each process uses the same seed, which is not desired
def generate_points_parallel(n):
     pool = mp.Pool()
     # we ask each process to generate n//mp.cpu_count() points
     return pool.map(count_inside_point, [n//mp.cpu_count()]*mp.cpu_count())

# set seed for each process
# first, let's define a helper function
def helper(args):
   n, s = args
   random.seed(s)
   return count_inside_point(n)
 
def generate_points_parallel_set_seed(n):
   pool = mp.Pool() # we can also specify the number of processes by Pool(number)
   # we ask each process to generate n//mp.cpu_count() points
   return pool.map(helper, list(zip([n//mp.cpu_count()]*mp.cpu_count(), range(mp.cpu_count())))) 

# another optimization via vectorization
def generate_points_vectorized(n):
    p = np.random.uniform(-1, 1, size=(n,2))
    return np.sum(np.linalg.norm(p, axis=1) <= 1)
 
def pi_naive(n):
    print(f'pi: {count_inside_point(n)/n*4:.6f}')
 
def pi_parallel(n):
    print(f'pi: {sum(generate_points_parallel_set_seed(n))/n*4:.6f}')
 
def pi_vectorized(n):
    print(f'pi: {generate_points_vectorized(n)/n*4:.6f}')
    

amount = 10**7

print(f'naive: {timeit.timeit("pi_naive(amount)", setup="from __main__ import pi_naive;from __main__ import amount",number=1):.6f} seconds\n')
print(f'parallel: {timeit.timeit("pi_parallel(amount)", setup="from __main__ import pi_parallel;from __main__ import amount",number=1):.6f} seconds\n')
print(f'vectorized: {timeit.timeit("pi_vectorized(amount)", setup="from __main__ import pi_vectorized;from __main__ import amount",number=1):.6f} seconds\n')
