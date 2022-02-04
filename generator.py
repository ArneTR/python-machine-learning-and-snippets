#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:41:45 2020

@author: light
"""


def fib():
    '''
    a generator to create Fibonacci numbers less than 10
    '''
    a, b = 0, 1
    while True:
        a, b = b, a+b
        yield a

f = fib()
print(f)
counter = 0
for e in fib(): 
    if counter >= 100: break 
    print(e)
    counter += 1
