#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:15:41 2018
@author: houben

This is a script to plot the groundwater head of a confined aquifer from a transient ogs simulation.

"""
import matplotlib.pyplot as plt
from ogs5py.reader import readtec_polyline
import numpy as np

tecs = readtec_polyline(
    task_id="transect_01",
    task_root="/Users/houben/PhD/transect/transect/ogs/confined/transient/topography/1layer/template/",
)

process = "GROUNDWATER_FLOW"
observation_point = "obs_0990"

head = tecs[process][observation_point]["HEAD"]
time = tecs[process][observation_point]["TIME"]
dist = tecs[process][observation_point]["DIST"]

# plot head at observation_poit over time
plt.plot(time, head[:, head.shape[1] - 1])


# calculate avaraged head
mean_head_per_timestep = []
mean_head_total = []
for values in tecs[process]:
    mean_head_per_timestep.append(tecs[process][values]["HEAD"][1, head.shape[1] - 1])
    mean_head_total.append(np.mean(mean_head_per_timestep))


# =============================================================================
# calculate averaged head per time over location for maximum value of each observation point
# =============================================================================
def average_head_time():
    """ NA """
    max_head_per_location = []
    mean_head_total = []
    for obs in tecs[process]:
        max_head_per_location.append(tecs[process][obs]["HEAD"][1, head.shape[1] - 1])
        mean_head_total.append(np.mean(max_head_per_location))
