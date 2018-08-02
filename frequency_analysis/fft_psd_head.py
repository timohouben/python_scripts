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
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.optimize as optimization

# =============================================================================
# global variables set manually
# =============================================================================
which_data_to_plot = 1 # 1: ogs, 2: gw_model 3: recharge
path_to_project = "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_HOMO_276_D/"
name_of_project_gw_model = "sinus"
name_of_project_ogs = "transect_01"
process = 'GROUNDWATER_FLOW'
which = 'max'       # min, max, mean
time_steps = 8400    # this is the value which is given in the ogs input file .tim. It will result in a total of 101 times because the initial time is added.
# variables for FFT
obs_point = ''
time_step_size = 86400

# =============================================================================
# global variables set automatically
# =============================================================================
print ("Reading .tec-files...")
tecs = readtec_polyline(task_id=name_of_project_ogs,task_root=path_to_project, single_file="/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_VERTICAL_276_D/transect_01_ply_obs_0200_t21_GROUNDWATER_FLOW.tec")
print ("Done.")

# =============================================================================
# get data dependent on which_data_to_plot
# =============================================================================
if which_data_to_plot == 1:
    head_ogs = gethead_ogs_each_obs(process, obs_point, which, time_steps, tecs=tecs, single_file=True)
    fft_data = head_ogs
    recharge = getrecharge(path_to_project=path_to_project, name_of_project_ogs=name_of_project_ogs, time_steps=time_steps)
elif which_data_to_plot == 2:
    head_gw_model = gethead_gw_model_each_obs(make_array_gw_model(
                        split_gw_model(getlist_gw_model(str(path_to_project) 
                        + str(name_of_project_gw_model) 
                        + '/H.OUT'), index=2)), convert_obs_list_to_index('obs_0990'))
    fft_data = head_gw_model
    recharge = getrecharge(path_to_project=path_to_project, name_of_project_ogs=name_of_project_ogs, time_steps=time_steps)
elif which_data_to_plot == 3:
    recharge = getrecharge(path_to_project=path_to_project, name_of_project_ogs=name_of_project_ogs, time_steps=time_steps)
    fft_data = recharge
 
# convert recharge from list to array    
recharge = np.asarray([float(i) for i in recharge])

# =============================================================================
# Calculate the discrete fourier transformation    
# =============================================================================

def fft_psd(fit=False, savefig=False):

    # employ a filter on the input data (head_ogs)
    # fft_data = signal.savgol_filter(fft_data, 19, 3)
    
    # generate frequencies for psd with scpfft (will not be used if Welche's Methos is applied)
    #freq = scpfft.fftfreq(len(fft_data), time_step_size)    
    #freq_month = freq * (30*86400)
    #ind=np.arange(1,len(fft_data)/2+1)

    # method 1
    # psd=2*abs(fft[ind])**2

    # method 2
    #psd=abs(fft[ind])**2

    # method 3
    #psd=2*abs(fft[ind])**2/len(fft_data)

    # method 4
    #psd=abs(fft[ind])**2/len(fft_data)

    # method 6 (by ESTANIS)
    #fft_norm = fft_data - np.mean(fft_data)
    #fft = scpfft.fft(fft_norm)
    #psd=abs(fft[ind])**2
    
    
    # some data porcessing:
    recharge_detrend = signal.detrend(recharge, type='linear')
    fft_data_detrend = signal.detrend(fft_data, type='linear')    
    
    # -------------------------------------------------------------------------
    # method 7: Welche's Method and division of output/input 
    #           (Jimiez-Martinez et al. 2013)
    # -------------------------------------------------------------------------


    # Step 1: define the sampling frequency
    # -------------------------------------------------------------------------
    # sampling_frequency = 3.80265176e-7  # sampling rate 1/month (~30.5 days)
    sampling_frequency = 0.00001157407  # sampling rate 1/day


    # Step 2: calculate the power spectrum of input (recharge) and output (head)
    #         and calculate abs(H_h(w))**2 with H_h(w) = Y(w) / X(w)
    # -------------------------------------------------------------------------
    frequency_input, power_spectrum_input = signal.welch(recharge_detrend, 
                                                        sampling_frequency, 
                                                        nperseg=500, 
                                                        window='hamming')
    frequency_output, power_spectrum_output = signal.welch(fft_data_detrend, 
                                                          sampling_frequency, 
                                                          nperseg=500, 
                                                          window='hamming')
    power_spectrum_result = abs((power_spectrum_output / power_spectrum_input))**2

    # Step 3: plot the resulting power spectrum
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1/month")
    #ax.set_ylim(1e-3,1e6)
    #ax.plot(freq_month[ind],psd)
    ax.plot(frequency_input, power_spectrum_result)
    ax.set_title('power spectruml density for observation point ' + str(obs_point))
    #ax.set_title('power spectruml density for recharge')
    ax.grid(color='grey', linestyle='--', linewidth=0.5, which='both')
    
    if fit == True:
        
        # Step 4: employ a filter on the spectrum to optimize the fit
        # ---------------------------------------------------------------------
        power_spectrum_result_filtered = signal.savgol_filter(power_spectrum_result, 9, 3)
        ax.plot(frequency_input, power_spectrum_result_filtered)
        
        # Step 5: least squares automatic fit for linear aquifer model (Gelhar, 1993):
        #          abs(H_h(w))**2 = 1 / (a**2 * ( 1 + ((t_l**2) * (w**2))))
        # ---------------------------------------------------------------------
        # make an initial guess for a_l, and t_l
        initial_guess = np.array([1e-15, 8640*30*2.6])

        # generate a weighing array.
        sigma = np.full((len(power_spectrum_result_filtered)),1.0)
        #sigma = np.arange(1,2,1/138.)
        #sigma = np.arange(1,10,9/138.)
        #sigma = np.flipud(np.arange(1,10,9/138.))

        # define the function to fit (linear aquifer model):
        def linear_fit(w, a, t_l):
            return 1 / (a**2 * ( 1 + ((t_l**2) * (w**2))))
        
        # perform the fit
        popt, pcov = optimization.curve_fit(linear_fit,
                                          frequency_input,
                                          power_spectrum_result_filtered,
                                          p0 = initial_guess)

        # print the resulting parameters
        print("Parameters for linear model: " + 
              "\na_l = " + 
              str(popt[0]) + 
              "\nt_l = " + 
              str(popt[1]))

        # manual or automatic fit (UNDER CONSTRUCTION)        
        a_l = 1e-15
        t_l = 8640*30*2.6
        a_l = popt[0]
        t_l = popt[1]

        # Step 6: Plot the linear fit model
        # ---------------------------------------------------------------------
        linear_model = []
        # fitting model for the linear reservoir (Gelhar, 1993)
        for i in range(0,len(frequency_input)):
            line = 1 / (a_l**2 * ( 1 + ((t_l**2) * (frequency_input[i]**2))))
            linear_model.append(line)
            
         # Step 5: least squares automatic fit for Dupuit-Aquifer model 
         # (e.g. Gelhar and Wilson, 1974): 
         # abs(H_h(w))**2 = (b/E)**2 * ( (1/O)*tanh)((1+j)*sqrt(1/2*O))*tanh((1-j)*sqrt(1/2*O))
         # O = td * w
         # E = x - x_o    distance from river
         # ---------------------------------------------------------------------
            
         !!! AUFGEHÖRT MIT DER IMPLEMENTIERUNG VON DUPUIT MODEL !!!
         

        ax.plot(frequency_input, linear_model)
        # calculate transmissivity and storage
        T = a_l * 1000**2 / 3
        kf = T/30.
        Ss = a_l * (t_l * 86400)
        #D = kf / Ss
        D = 1000**2 / (3. * (t_l * 86400))
        print('T [m2/s]: ' + str(T)  + '\n' +
              'Ss [1/m]: ' + str(Ss) + '\n' +
              'kf [m/s]: ' + str(kf) + '\n' +
              'D [m2/s]: ' + str(D) + '\n' +
              'a : ' + str(a_l)+ '\n' +
              't_c : ' + str(t_l)
              )
        '''
        plt.text(min(frequency_input),min(psd), 'T [m2/s]: ' + str(T)  + '\n' +
                 'Ss [1/m]: ' + str(Ss) + '\n' +
                 'kf [m/s]: ' + str(kf) + '\n' +
                 'D [m2/s]: ' + str(D) + '\n' +
                 'a : ' + str(a_l)+ '\n' +
                 't_c : ' + str(t_l)
                 )
        '''
        
    if savefig == True:
        fig.savefig(str(path_to_project) + 'FREQUENCY_7_' + str(os.path.basename(str(path_to_project)[:-1])) + '_' + str(obs_point) + ".png")
    #fig.savefig(str(obs_point) + '_' + str(time.strftime("%Y%m%d%H%M%S")))
  