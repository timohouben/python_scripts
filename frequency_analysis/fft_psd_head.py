#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:21:48 2018

To Improve:
    - not functioning for gw_head!!!
"""

# =============================================================================
# import modules
# =============================================================================

# import sys and set path to the module
import sys
sys.path.append("/Users/houben/PhD/python/scripts/head_ogs_vs_gw-model/transient")
from conf_head_ogs_vs_gw_model_trans import gethead_ogs_each_obs, getrecharge, gethead_gw_model_each_obs, make_array_gw_model, split_gw_model, getlist_gw_model, convert_obs_list_to_index
from ogs5py.reader import readtec_polyline
import scipy.fftpack as scpfft
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# =============================================================================
# global variables set manually
# =============================================================================
which_data_to_plot = 1 # 1: ogs, 2: gw_model 3: recharge
path_to_project = "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_C_month/Groundwater@UFZ_eve_HOMO_276_C/"
name_of_project_gw_model = "sinus"
name_of_project_ogs = "transect_01"
process = 'GROUNDWATER_FLOW'
which = 'max'       # min, max, mean
time_steps = 276    # this is the value which is given in the ogs input file .tim. It will result in a total of 101 times because the initial time is added.
# variables for FFT
obs_point = 'obs_0200'
time_step_size = 86400 * 30

# =============================================================================
# global variables set automatically
# =============================================================================

tecs = readtec_polyline(task_id=name_of_project_ogs,task_root=path_to_project)

# =============================================================================
# get data dependent on which_data_to_plot
# =============================================================================
if which_data_to_plot == 1:
    head_ogs = gethead_ogs_each_obs(process, obs_point, which, time_steps)
    fft_data = head_ogs
elif which_data_to_plot == 2:
    head_gw_model = gethead_gw_model_each_obs(make_array_gw_model(
                        split_gw_model(getlist_gw_model(str(path_to_project) 
                        + str(name_of_project_gw_model) 
                        + '/H.OUT'), index=2)), convert_obs_list_to_index('obs_0990'))
    fft_data = head_gw_model
elif which_data_to_plot == 3:
    recharge = getrecharge(mm_m=True)
    fft_data = recharge

# =============================================================================
# Calculate the discrete fourier transformation    
# =============================================================================

def fft_psd(fft_data=fft_data, time_step_size=time_step_size, fit=False, a=0, t_c=0):

    fft = scpfft.fft(fft_data)
    freq = scpfft.fftfreq(len(fft_data), time_step_size)    
    freq_month = freq * (30*86400)
    ind=np.arange(1,len(fft_data)/2+1)

    # method 1
    # psd=2*abs(fft[ind])**2

    # method 2
    #psd=abs(fft[ind])**2

    # method 3
    #psd=2*abs(fft[ind])**2/len(fft_data)

    # method 4
    #psd=abs(fft[ind])**2/len(fft_data)

    # method 6 (by ESTANIS)
    fft_data_norm = fft_data - np.mean(fft_data)
    fft = scpfft.fft(fft_data_norm)
    psd=abs(fft[ind])**2

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1/month")
    #ax.set_ylim(1e-3,1e6)
    ax.plot(freq_month[ind],psd)
    
    ax.set_title('power spectral density for observation point ' + str(obs_point))
    #ax.set_title('power spectral density for recharge')
    
    ax.grid(color='grey', linestyle='--', linewidth=0.5, which='both')
    
    if fit == True:

        # fitting model for the linear reservoir (Gelhar, 1993)
        #####-----some changes here ----- #####
        a = 0.1
        t_c = 20
        f_w = []

        for i in range(0,len(freq_month[ind])):
            line = 1 / (a * ( 1 + ((t_c**2) * (freq_month[i]**2))))
            f_w.append(line)

        ax.plot(freq_month[ind],f_w)
        # calculate transmissivity and storage
        T = a * 1000**2 / 3
        kf = T/30.
        Ss = a * (t_c * 86400 * 30)
        #D = kf / Ss
        D = 1000**2 / (3. * (t_c * 86400 * 30.))
        print('T [m2/s]: ' + str(T)  + '\n' +
              'Ss [1/m]: ' + str(Ss) + '\n' +
              'kf [m/s]: ' + str(kf) + '\n' +
              'D [m2/s]: ' + str(D) + '\n' +
              'a : ' + str(a)+ '\n' +
              't_c : ' + str(t_c)
              )
        plt.text(min(freq_month),min(psd), 'T [m2/s]: ' + str(T)  + '\n' +
                 'Ss [1/m]: ' + str(Ss) + '\n' +
                 'kf [m/s]: ' + str(kf) + '\n' +
                 'D [m2/s]: ' + str(D) + '\n' +
                 'a : ' + str(a)+ '\n' +
                 't_c : ' + str(t_c)
                 )


    
    fig.savefig(str(path_to_project) + 'FREQUENCY_6_RECHARGE' + str(os.path.basename(str(path_to_project)[:-1])) + '_' + str(obs_point) + ".png")
    #fig.savefig(str(obs_point) + '_' + str(time.strftime("%Y%m%d%H%M%S")))