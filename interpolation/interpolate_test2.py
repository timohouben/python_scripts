#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:09:04 2018

@author: houben
"""
import numpy as np
from scipy.interpolate import interp1d

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x ** 2 / 9.0)
f = interp1d(x, y)
f2 = interp1d(x, y, kind="cubic")

xnew = np.linspace(0, 10, num=41, endpoint=True)
import matplotlib.pyplot as plt

plt.plot(x, y, "o", xnew, f(xnew), "-", xnew, f2(xnew), "--")
plt.legend(["data", "linear", "cubic"], loc="best")
plt.show()
