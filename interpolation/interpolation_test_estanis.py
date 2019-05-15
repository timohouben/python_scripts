#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:49:13 2018

@author: houben
"""

import numpy as np
from pandas import datetime
import pandas
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

array = np.random.rand(1000, 2)
array[:, 0].sort()
# df = pandas.DataFrame(array)
# interpolate = df.interpolate(method='time', axis=1)

d = interp1d(array[:, 0], array[:, 1])
d2 = interp1d(array[:, 0], array[:, 1], kind="cubic")


xnew = np.arange(min(array[:, 0]), max(array[:, 0]), 0.001)

plt.figure()
plt.plot(array[:, 0], array[:, 1])
plt.plot(xnew, d(xnew))
test = np.column_stack((xnew, d(xnew)))
plt.plot(test[:, 0], test[:, 1])
