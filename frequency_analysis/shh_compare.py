#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
compare shh analytisch und ogs run
"""
import sys
import numpy as np
sys.path.append("/Users/houben/PhD/python/scripts/frequency_analysis")
from fft_psd_head import fft_psd
import matplotlib.pyplot as plt
import time
from shh_analytical import shh_analytical
then = time.time()


fft_data = np.loadtxt('/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/frequency/dupuit_flow/D_30_1000_30_whitenoise_D_18-D_30_homogeneous/head_ogs_obs_0400_mean.txt')
recharge = np.loadtxt('/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/frequency/dupuit_flow/D_30_1000_30_whitenoise_D_18-D_30_homogeneous/rfd_curve#1.txt')

path_to_project = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/frequency/dupuit_flow/D_30_1000_30_whitenoise_D_18-D_30_homogeneous"
method = "scipyffthalf"
obs_point = "obs_0400"
single_file_name = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/frequency/dupuit_flow/D_30_1000_30_whitenoise_D_18-D_30_homogeneous"
path_to_multiple_projects = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/frequency/dupuit_flow/D_30_1000_30_whitenoise_D_18-D_30_homogeneous"
aquifer_thickness = 30
aquifer_length = 1000
obs_point_list = ['obs_0000', 'obs_0010', 'obs_0100', 'obs_0200', 'obs_0300', 'obs_0400', 'obs_0500', 'obs_0600', 'obs_0700', 'obs_0800', 'obs_0900', 'obs_0950', 'obs_0960', 'obs_0970', 'obs_0980', 'obs_0990', 'obs_1000']
distance_to_river_list = [1000, 990, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 40, 30, 20, 10, 0.1]
distance_to_river = 600
time_steps = 8401
time_step_size = 86400
threshold=1

                      
T_l, kf_l, Ss_l, D_l, a_l, t_l, T_d, kf_d, Ss_d, D_d, a_d, t_d, power_spectrum_output, power_spectrum_input, power_spectrum_output, power_spectrum_result, frequency_input = fft_psd(
                                            fft_data=fft_data, 
                                            recharge=recharge,
                                            #threshold=thresholds[j],
                                            #threshold=threshold,
                                            path_to_project=path_to_project, 
                                            method=method, 
                                            saveoutput=False, obs_point=obs_point, 
                                            single_file=path_to_project+"/"+single_file_name, 
                                            dupuit=False,
                                            aquifer_thickness=aquifer_thickness,
                                            aquifer_length=aquifer_length,
                                            time_step_size=86400
                                            )



Sy = 9e-5
T = 30*(1e-5)
x = 400
power_spectrum_anal = shh_analytical(power_spectrum_input, frequency_input, Sy, T, x, aquifer_length, m=2, n=2)

plt.loglog(frequency_input, power_spectrum_output)
plt.loglog(frequency_input, power_spectrum_anal)
plt.show()
