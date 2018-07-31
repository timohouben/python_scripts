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
which_data_to_plot = 2 # 1: ogs vs gw_model, 2: ogs, 3: gw_model
path_to_project = "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/"
name_of_project_gw_model = "sinus"
name_of_project_ogs = "transect_01"
process = 'GROUNDWATER_FLOW'
which = 'all'       # min, max, mean
time_steps = 10   # this is the value which is given in the ogs input file .tim. It will result in a total of time_steps+1 times because the initial time is added.
obs_per_plot =  ['obs_0200', 'obs_0800', 'obs_0950', 'obs_0990']

#['obs_0100', 'obs_0200', 'obs_0300', 'obs_0400', 'obs_0500', 'obs_0600', 'obs_0700', 'obs_0800', 'obs_0900'] 