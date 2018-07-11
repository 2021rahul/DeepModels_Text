# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:31:12 2018

@author: rahul.ghosh
"""

import numpy as np
import random


def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    r = random.random()
    total = 0.0
    for i in range(len(a)):
        total += a[i]
        if total > r:
            return i
    return len(a)-1
