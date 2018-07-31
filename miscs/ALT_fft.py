#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:21:48 2018

@author: houben
"""

# =============================================================================
# import modules
# =============================================================================
from conf_head_ogs_vs_gw_model_trans import gethead_ogs_each_obs, getrecharge
from ogs5py.reader import readtec_polyline
import scipy.fftpack as scpfft
import numpy as np
import matplotlib.pyplot as plt
import math

# =============================================================================
# global variables set manually
# =============================================================================
which_data_to_plot = 1 # 1: ogs vs gw_model, 2: ogs, 3: gw_model
path_to_project = "/Users/houben/PhD/transect/transect/ogs/confined/transient/topography/1layer/template_40m/"
name_of_project_gw_model = "sinus"
name_of_project_ogs = "transect_01"
process = 'GROUNDWATER_FLOW'
which = 'mean'       # min, max, mean
time_steps = 100    # this is the value which is given in the ogs input file .tim. It will result in a total of 101 times because the initial time is added.
obs_per_plot = ['obs_0000', 'obs_0950', 'obs_0960', 'obs_0970', 'obs_0980', 'obs_0990']

# variables for FFT
obs_point = 'obs_0950'

# =============================================================================
# global variables set automatically
# =============================================================================

recharge = []
recharge_mm_d = []
tecs = readtec_polyline(task_id=name_of_project_ogs,task_root=path_to_project)
time_s = tecs[process][obs_per_plot[0]]["TIME"]
time_d = time_s / 86400


# =============================================================================
# Calculate the discrete fourier transformation    
# =============================================================================

recharge = getrecharge(mm_d=True)
    
#head_gw_model = gethead_gw_model_each_obs(make_array_gw_model(
#                        split_gw_model(getlist_gw_model(str(path_to_project) 
#                        + str(name_of_project_gw_model) 
#                        + '/H.OUT'), index=2)), convert_obs_list_to_index('obs_0990'))

head_ogs = gethead_ogs_each_obs(process, obs_point, which, time_steps)
 
''' 
#############
# script to visualize the 

# generate sample data
x = np.arange(0, math.pi*50, 0.01)
sin = np.sin(x) + np.sin(2*x) + np.sin(3*x)*0.5# + np.sin(15*x) + np.sin(150*x)*0.5
sin1 = np.sin(x)
sin2 = np.sin(2*x)
sin3 = np.sin(x)*0.5

time_step_size = 86400

fft = scpfft.fft(sin)
freq = scpfft.fftfreq(len(sin), time_step_size)    
ind=np.arange(1,len(sin)/2+1)
psd=abs(fft[ind])**2#+abs(fft[-ind])**2


fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(4,2,1)
ax.plot(x[:2000],sin[:2000], label='1', color='b')
ax.set_ylabel('signal')

ax = fig.add_subplot(4,2,3)
ax.plot(x[:2000],sin1[:2000], color='green')
ax.set_ylabel('sin(x)')

ax = fig.add_subplot(4,2,5)
ax.plot(x[:2000],sin2[:2000], color='orange')
ax.set_ylabel('sin(2x)')

ax = fig.add_subplot(4,2,7)
ax.plot(x[:2000],sin3[:2000], label='1', color='b')
ax.set_ylabel('1/2 * sin(x)')

ax = fig.add_subplot(1,2,2)
ax.plot(freq[ind],psd)
ax.set_title('power spec. density')
#################
'''

#freq = fft.fftfreq(101, 86400)    
#fft = fft.fft(head_ogs)
#ind=np.arange(1,101/2+1)   
#psd=abs(fft[ind])**2+abs(fft[-ind])**2



fig = plt.figure(figsize=(12, 5))
plt.suptitle('Frequency analysis for ' + str(obs_point) + ' from ogs results.', fontsize=16)

ax = fig.add_subplot(1,4,1)
ax.plot(np.arange(0,time_steps+1,1),recharge)
ax.set_title('recharge [mm/d]')   

ax = fig.add_subplot(1,4,2)
ax.plot(head_ogs)   
ax.set_title('head_ogs [m]')

ax = fig.add_subplot(1,4,3)
ax.set_xscale("log")
ax.plot(freq[ind],psd)
ax.set_title('power spectral density')

ax = fig.add_subplot(1,4,4)
ax.set_xscale("log")
ax.plot(freq[ind],fft[ind])
ax.set_title('Fast Frourier Transformation')
    
    
    
    #signal = head_ogs - head_ogs.mean() # don't know y this step: ask Estanis
    #fourier = rfft(signal)
    #n = len(signal)
    #timestep = 1000000
    #freq = fftfreq(n, timestep)
    ##plt.plot(freq,fourier)

    #print(np.c_[freq,abs(signal)])
    #plt.plot(freq, abs(signal))
    