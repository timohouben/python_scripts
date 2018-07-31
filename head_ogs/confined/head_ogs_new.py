#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:01:22 2018

@author: houben
"""

# =============================================================================
# import modules
# =============================================================================
import matplotlib.pyplot as plt
from ogs5py.reader import readtec_polyline
from scipy.fftpack import rfft, fftfreq
import numpy as np
import re
import os

# =============================================================================
# global variables set manually
# =============================================================================
path_to_project = "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/"
name_of_project_ogs = "transect_01"
process = 'GROUNDWATER_FLOW'
which = 'all'       # min, max, mean
time_step = 1    # select the time step you want to plot: 0: inital conditions, if steady state: 1



tecs = readtec_polyline(task_id=name_of_project_ogs,task_root=path_to_project)

for tecs[process] ### noch nicht weiter gemacht
head_ogs_per_time_step = tecs[process][observation_point]["HEAD"][:,number_of_columns-1]