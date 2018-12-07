#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:05:26 2018

@author: houben
"""

import numpy as np
import scipy.optimize as optimization
import matplotlib.pyplot as plt
from scipy import integrate

path_to_folder = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30_mHM/"
# variables as locations

model_runs = ["Groundwater@UFZ_eve_HOMO_276_D_18",
               "Groundwater@UFZ_eve_HOMO_276_D_19",
               "Groundwater@UFZ_eve_HOMO_276_D_20",
               "Groundwater@UFZ_eve_HOMO_276_D_21",
               "Groundwater@UFZ_eve_HOMO_276_D_22",
               "Groundwater@UFZ_eve_HOMO_276_D_23",
               "Groundwater@UFZ_eve_HOMO_276_D_24",
               "Groundwater@UFZ_eve_HOMO_276_D_25",
               "Groundwater@UFZ_eve_HOMO_276_D_26",
               "Groundwater@UFZ_eve_HOMO_276_D_27",
               "Groundwater@UFZ_eve_HOMO_276_D_28",
               "Groundwater@UFZ_eve_HOMO_276_D_29",
               "Groundwater@UFZ_eve_HOMO_276_D_30"]

boundary_cond = 30

for j in model_runs:
    obs_0000 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0000_mean.txt')
    obs_0010 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0010_mean.txt')
    obs_0100 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0100_mean.txt')
    obs_0200 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0200_mean.txt')
    obs_0300 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0300_mean.txt')
    obs_0400 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0400_mean.txt')
    obs_0500 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0500_mean.txt')
    obs_0600 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0600_mean.txt')
    obs_0700 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0700_mean.txt')
    obs_0800 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0800_mean.txt')
    obs_0900 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0900_mean.txt')
    obs_0950 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0950_mean.txt')
    obs_0960 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0960_mean.txt')
    obs_0970 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0970_mean.txt')
    obs_0980 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0980_mean.txt')
    obs_0990 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_0990_mean.txt')
    obs_1000 = np.loadtxt(path_to_folder + j + '/head_ogs_obs_1000_mean.txt')
    
    loc = np.asarray([0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 960, 970, 980, 990, 1000])
    average_head = []
    head = np.zeros((17))
    length = 1000
    for i in range(0,len(obs_0000)):
        # make an array with head for locations at timestep i
        head[0] = obs_0000[i] - boundary_cond
        head[1] = obs_0010[i] - boundary_cond
        head[2] = obs_0100[i] - boundary_cond
        head[3] = obs_0200[i] - boundary_cond
        head[4] = obs_0300[i] - boundary_cond
        head[5] = obs_0400[i] - boundary_cond
        head[6] = obs_0500[i] - boundary_cond
        head[7] = obs_0600[i] - boundary_cond
        head[8] = obs_0700[i] - boundary_cond
        head[9] = obs_0800[i] - boundary_cond
        head[10] = obs_0900[i] - boundary_cond
        head[11] = obs_0950[i] - boundary_cond
        head[12] = obs_0960[i] - boundary_cond
        head[13] = obs_0970[i] - boundary_cond
        head[14] = obs_0980[i] - boundary_cond
        head[15] = obs_0990[i] - boundary_cond
        head[16] = obs_1000[i] - boundary_cond
        integral = integrate.simps(head,loc)
        average_head.append(integral / length)
    plt.plot(average_head, label='average head')
    plt.legend()

    average_head[-1]=average_head[-2]
    print("saving " + path_to_folder + j + '/average_head.txt')
    np.savetxt(path_to_folder + j + '/average_head_-BC.txt',average_head)












'''
p0 = [1]
def head_func(x, a):
    #return a * x**2 + b*x + c)**(1/2)
    return (a * 1000**2 * (1 - x**2 / 1000**2) + head[0])
popt, pcov = optimization.curve_fit(head_func, loc, head, p0=p0)
plt.plot(loc,[head_func(j, popt[0]) for j in loc], label='fit')
plt.plot(loc, head, label='data')
plt.legend()

p0 = [0.001]
def head_func(x, a):
    #return a * x**2 + b*x + c)**(1/2)
    return np.sqrt(head[0]**2 + (head[16]**2 - head[0]**2) * x / 1000 + a * x * (1000 - x))
popt, pcov = optimization.curve_fit(head_func, loc, head, p0=p0)
plt.plot(loc,[head_func(j, popt[0]) for j in loc], label='fit')
plt.plot(loc, head, label='data')
plt.legend()
 

p0 = [1,1,1,1]
def head_func(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x +d
popt, pcov = optimization.curve_fit(head_func, loc, head, p0=p0)
plt.plot(loc,[head_func(j, popt[0], popt[1], popt[2], popt[3]) for j in loc], label='fit')
plt.plot(loc, head, label='data')
plt.legend()

p0 = [1,1]
def head_func(x, a, b,):
    return (a*x + b*x)**(1/2)
popt, pcov = optimization.curve_fit(head_func, loc, head, p0=p0)
plt.plot(loc,[head_func(j, popt[0], popt[1]) for j in loc], label='fit')
plt.plot(loc, head, label='data')
plt.legend()
'''