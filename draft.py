# -*- coding: utf-8 -*-
"""
Author: VincentGum
"""
import pandas as pd
import numpy as np
import EM
data = pd.read_csv('data/Q2Q3_input.csv')
data.pop('user_id')

# set the initial clusters' centers like this
c1 = np.array([1, 1, 1, 1, 1, 1])
c2 = np.array([0, 0, 0, 0, 0, 0])
c_init = [c1, c2]

em = EM.EM(data.values, c_init)
em.train()

print(em.SSE)



