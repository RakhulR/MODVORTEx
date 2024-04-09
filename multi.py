# -*- coding: utf-8 -*-
"""
Created on Sat May 20 09:11:45 2023

@author: Rakhul Raj
"""

from multiprocessing import Pool
from functools import partial

def square(x, y):
    return x * x + y

if __name__ == '__main__':
    numbers = [1, 2, 3]
    second_arg = 2
    
    square_partial = partial(square, y=second_arg)
    with Pool(4) as p:
        
        results = p.map(square_partial, numbers)
        print(results)