#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:06:08 2019

@author: houben

script to generate multiple Shh anaytical with different transmissivities and storativities
"""

from shh_analytical import shh_analytical
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

T_list = [1e-2, 1e-3, 1e-4, 1e-5]
S_list = [0.5, 0.4, 0.3, 0.2, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
length_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
sww = np.loadtxt("/Users/houben/Desktop/shh_analytical_multi/sww.txt")
frequency = np.loadtxt("/Users/houben/Desktop/shh_analytical_multi/frequency.txt")
directory = "/Users/houben/Desktop/shh_analytical_multi"

X = (frequency, sww)

for T in T_list:
    for S in S_list:
        for L in length_list:
            for x in np.linspace(0, L, 5):
                shh = shh_analytical(X, S, T, x, L, m=5, n=5, norm=False)
                plt.title(
                    "Transmissivity: "
                    + str(T)
                    + "\n"
                    + "Storativity: "
                    + str(S)
                    + "\n"
                    + "Length: "
                    + str(L)
                    + "\n"
                    + "Location: "
                    + str(x)
                )
                plt.loglog(frequency, shh)
                plt.ylim(10e-7, 10e13)
                plt.savefig(
                    directory
                    + "/"
                    + str(T)
                    + "_"
                    + str(S)
                    + "_"
                    + str(L)
                    + "_"
                    + str(x)
                    + ".png"
                )
                print(
                    "saving "
                    + str(T)
                    + "_"
                    + str(S)
                    + "_"
                    + str(L)
                    + "_"
                    + str(x)
                    + ".png"
                )
                plt.close()
