#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 09:54:44 2018

@author: houben

This is a smal script to execute the psd_fft_head script multiple times for various model results and observation points.
"""

import sys
import numpy as np
sys.path.append("/Users/houben/PhD/python/scripts/frequency_analysis")
from fft_psd_head import fft_psd, get_fft_data_from_simulation
import matplotlib.pyplot as plt

'''
methods = ['scipyfftnormt', 'scipyfftnormn', 'scipyfft', 'scipywelch',
           'pyplotwelch', 'scipyperio', 'spectrumperio']
'''

methods = ['scipyfft']

'''
# configuration for homogeneous model runs
path_to_project_list = ["/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_1_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_2_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_3_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_4_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_5_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_6_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_7_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_8_results/"]                      
single_file_names = ["transect_01_ply_obs_0100_t3_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0300_t5_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0400_t6_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0600_t8_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0800_t10_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0950_t12_GROUNDWATER_FLOW.tec"]
obs_point_list = ["0100", "0300", "0400", "0600", "0800", "0950"]
'''                   
'''
# configuration for homo/verticak/horizontal
path_to_project_list = ["/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_HOMO_276_D/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_HORIZONTAL_276_D/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_VERTICAL_276_D/"]
single_file_names = ["transect_01_ply_obs_0100_t11_GROUNDWATER_FLOW.tec",
                     "transect_01_ply_obs_0300_t31_GROUNDWATER_FLOW.tec",
                     "transect_01_ply_obs_0400_t41_GROUNDWATER_FLOW.tec",
                     "transect_01_ply_obs_0600_t61_GROUNDWATER_FLOW.tec",
                     "transect_01_ply_obs_0800_t81_GROUNDWATER_FLOW.tec",
                     "transect_01_ply_obs_0950_t96_GROUNDWATER_FLOW.tec"]
obs_point_list = ["0100", "0300", "0400", "0600", "0800", "0950"]
'''

# configuration for homogeneous models, 2nd run
path_to_project_list = ["/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_18_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_19_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_20_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_21_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_22_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_23_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_24_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_25_results/",                      
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_26_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_27_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_28_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_29_results/",
                        "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_30_results/"]
name_of_model_runs = ["D_18", "D_19", "D_20", "D_21", "D_22", "D_23", "D_24", "D_25", "D_26", "D_27", "D_28", "D_29", "D_30"]
single_file_names = ["transect_01_ply_obs_0000_t1_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0200_t4_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0400_t6_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0500_t7_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0600_t8_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0800_t10_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0950_t12_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_0990_t16_GROUNDWATER_FLOW.tec",
                    "transect_01_ply_obs_1000_t17_GROUNDWATER_FLOW.tec"]
obs_point_list = ["0000", "0200", "0400", "0500", "0600", "0800", "0950", "0990", "1000"]
distance_to_river_list = [0, 800, 600, 500, 400, 200, 50, 10, 0.01]
thresholds = [3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 1e-7, 2e-7, 8e-7]
path_to_results="/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/fitting_results/"
  

obs_index = 0 # index for iterating through the observation points
i=0 # index for different model runs
j=0 # index for different observation points
k=0 # index for different methods, PSD

T_l = np.zeros((len(path_to_project_list),len(single_file_names), len(methods)))
kf_l = np.zeros_like(T_l)
Ss_l = np.zeros_like(T_l)
D_l = np.zeros_like(T_l)
a_l = np.zeros_like(T_l)
t_l = np.zeros_like(T_l)
T_d = np.zeros((len(path_to_project_list),len(single_file_names), len(methods)))
kf_d = np.zeros_like(T_l)
Ss_d = np.zeros_like(T_l)
D_d = np.zeros_like(T_l)
a_d = np.zeros_like(T_l)
t_d = np.zeros_like(T_l)

for i,path_to_project in enumerate(path_to_project_list):
    obs_index = 0          
    for j,single_file_name in enumerate(single_file_names):
        print(path_to_project+single_file_name)
        fft_data, recharge = get_fft_data_from_simulation(path_to_project=path_to_project,
                                     single_file=path_to_project+single_file_name)
        #fft_data = fft_data[0:] # eliminate effects from wrong initial conditions
        #recharge = recharge[0:] # eliminate effects from wrong initial conditions
        obs_point = obs_point_list[obs_index]
        for k,method in enumerate(methods):
            
            T_l[i,j,k], kf_l[i,j,k], Ss_l[i,j,k], D_l[i,j,k], a_l[i,j,k], t_l[i,j,k], T_d[i,j,k], kf_d[i,j,k], Ss_d[i,j,k], D_d[i,j,k], a_d[i,j,k], t_d[i,j,k] = fft_psd(
                                            fft_data=fft_data, 
                                            recharge=recharge,
                                            #threshold=thresholds[obs_index],
                                            threshold=1e-6,
                                            path_to_project=path_to_project, 
                                            method=method, 
                                            fit=True, savefig=True, 
                                            saveoutput=True, obs_point=obs_point, 
                                            single_file=path_to_project+single_file_name, 
                                            aquifer_thickness=30, aquifer_length=1000, 
                                            dupuit=True,
                                            distance_to_river=distance_to_river_list[obs_index])
        obs_index += 1

    #if i == 1:
     #   break

print('Created all PSDs. Continue with plotting of results from linear fitting...')

params_l = [T_l, kf_l, Ss_l, D_l, a_l, t_l]
params_d = [T_d, kf_d, Ss_d, D_d, a_d, t_d]
labels = ["T", "kf", "Ss", "D", "a", "t"]


def plot(method=method, params_l=params_l, params_d=params_d, labels=labels):
    # linear model
    i=0 # index for different model runs
    j=0 # index for different LABELS
    k=0 # index for different methods, PSD            
    plt.ioff()
    for j,param in enumerate(params_l):
        for k,method in enumerate(methods):
            for i,path_to_project in enumerate(path_to_project_list):
                plt.title('Method: ' + str(method) + '\nFit: "linear"' + '\nParameter: ' + str(labels[j]))
                plt.grid(True)
                plt.xlabel('observation point')
                plt.ylabel(str(labels[j]))
                # configuration for homogeneous model runs
                #plt.semilogy(obs_point_list, param[i,:,k], label=path_to_project[-12:-9])
                # configuration for homo/vertical/horizontal
                plt.semilogy(obs_point_list, param[i,:,k], label=name_of_model_runs[i])
                plt.legend(loc='best')
            plt.savefig(str(path_to_results) + str(labels[j]) + '_' + str(method) + '_linear' + '.png')
            plt.close()
            
    # Dupuit model
    i=0 # index for different model runs
    j=0 # index for different LABELS
    k=0 # index for different methods, PSD            
    plt.ioff()
    for j,param in enumerate(params_d):
        for k,method in enumerate(methods):
            for i,path_to_project in enumerate(path_to_project_list):
                plt.title('Method: ' + str(method) + '\nFit: "Dupuit"' + '\nParameter: ' + str(labels[j]))
                plt.grid(True)
                plt.xlabel('observation point')
                plt.ylabel(str(labels[j]))
                # configuration for homogeneous model runs
                #plt.semilogy(obs_point_list, param[i,:,k], label=path_to_project[-12:-9])
                # configuration for homo/vertical/horizontal
                plt.semilogy(obs_point_list, param[i,:,k], label=name_of_model_runs[i])
                plt.legend(loc='best')
            plt.savefig(str(path_to_results) + str(labels[j]) + '_' + str(method) + '_dupuit' + '.png')
            plt.close()            
    print('Fertig!')

plot()