m#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
########################################
#   the following must be considered:
# - initial conditions in OGS must be switched off (saturation will be = 1 in whole domain)
# - script not yet functioning for transient analysis
# - 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import os
import numpy as np
import re
import glob
import time
then = time.time()
cwd = os.getcwd()

# =============================================================================
# extract head over x for 1 time step from OGS TECPLOT from vertical observation lines
# =============================================================================
#### grab all files ending with .tec
tec_files = glob.glob('*.tec')
#### sort the list with file names
tec_files.sort()

#### Defne some useful globar variables
number_obs = sum(1 for line in open("H.OUT", "r")) - 2
obs = np.linspace(0, 1000, 101)
timesteps = 0
z_fit = []
error = 0
#number_rows = 0
file_name_ogs = tec_files[0]
rmse_anal_numeric = 0

#### extracts the number of time steps used (inkl. initial condition)
data_H_ogs = open(file_name_ogs,'r')
for i, line in enumerate(data_H_ogs):
    if line[0].isdigit() != True:
        timesteps = timesteps + 1
timesteps = timesteps/3

#### generate array for head at obs
head_ogs = np.zeros([number_obs, timesteps])
head_gw_model = []

#### generate x-z-data for plots
for k, file_name_ogs in enumerate(tec_files):
    sat_z_data_ogs = []
    data_H_ogs = open(file_name_ogs,'r')
    #### extracts the number of time steps used (inkl. initial condition)
    for i, line in enumerate(data_H_ogs):
        if line[0].isdigit() == True:
            line_numbers = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
            sat_z_data_ogs.append(line_numbers)
    #number_rows = (i + 1 - timesteps * 2) / 2                                       # number of rows each time step

    #### find position of decreasing saturation (first time sat != 1)
    for j, x in enumerate(sat_z_data_ogs):
        try:
            if float(sat_z_data_ogs[j][1]) < 1:                                     # configutation for PRESSURE1 as output
            #if sat_z_data_ogs[j][1] > sat_z_data_ogs[j+1][1]: # and sat_z_data_ogs[j+1][1] > sat_z_data_ogs[j+2][1] and sat_z_data_ogs[j+2][1] > sat_z_data_ogs[j+3][1]:# and sat_z_data_ogs[j+3][1] > sat_z_data_ogs[j+4][1] and sat_z_data_ogs[j+4][1] > sat_z_data_ogs[j+5][1]:             # configuration for SATURATION1 as output
                head_ogs[k, 0] = sat_z_data_ogs[j][0]        
                break
        except: IndexError
data_H_ogs.close()

# =============================================================================
# polynomial fit of ogs data
# =============================================================================
'''
def poly_fit(x):
    fit_coeff = np.polyfit(obs, head_ogs[:,0], 4)
    z = fit_coeff[0]* x**2 + fit_coeff[1]* x**2 + fit_coeff[2] * x**2 + fit_coeff[3] * x + fit_coeff[4]
    return z

for i, line in enumerate(obs):
    line = poly_fit(obs[i])
    z_fit.append(line)
'''    




# =============================================================================
# extract head over x for 1 time step from gw_model deRooij 2012
# =============================================================================

data_H_gw_model = open("H.out","r")

def getlist(data_H_gw_model):
    head_x_data_gw_model = []
    for line in data_H_gw_model:
        line_numbers = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
        head_x_data_gw_model.append(line_numbers)
    del head_x_data_gw_model[0:2]
    data_H_gw_model.close()
    return head_x_data_gw_model

def split(data):
    for i in range(0,len(data)):
        head_gw_model.append(float(data[i][2]))

# =============================================================================
# calculate RMSE
# =============================================================================

def error(head_ogs, head_gw_model):
    rmse = np.sqrt(((head_ogs[:,0] - head_gw_model) ** 2).mean())
    return rmse

# =============================================================================
# plot data
# =============================================================================

def plot():
        #fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(15, 10), subplot_kw={'projection': '3d'})
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('Head Analytical vs Head Numerical\nsteady state - unconfined\n'+str(cwd))
    
        ax = fig.add_subplot(1, 2, 1)
        ax.set_ylim(30, 36)
        ax.set_xlim(0, 1010)
        plt.xticks(np.arange(0,1100,100))
        plt.yticks(np.arange(30,36.2,0.2))
        #ax.text(obs[0], ((max(head_ogs[:,0]) - min(head_ogs[:,0]))*0.1 + min(head_ogs[:,0])), "RMSE: " + str(rmse) + " m")
        #ax.text(obs[0], ((max(head_ogs[:,0]) - min(head_ogs[:,0]))*0.15 + min(head_ogs[:,0])), "Mean groundwater head: " + str(mean_head) + " m")
        ax.text(obs[10], ((max(head_ogs[:,0]) - min(head_ogs[:,0]))*0.1 + min(head_ogs[:,0])), "RMSE Anal - Num: " + str(format(rmse, "10.2E")) + " m")
        ax.text(obs[10], ((max(head_ogs[:,0]) - min(head_ogs[:,0]))*0.15 + min(head_ogs[:,0])), "Mean groundwater head: " + str(round(mean_head,2)) + " m")
        
        ax.plot(obs, head_gw_model, label="head analytical [m]")
        ax.plot(obs, head_ogs[:, 0], label="head numerical [m]")
        #ax3 = ax.plot(obs, z_fit)
        ax.set_title("head vs location")
        ax.set_xlabel('location [m]')
        ax.set_ylabel('head [m]')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        
        ax = fig.add_subplot(1, 2, 2)
        ax.set_xlim(min(head_ogs[:,0]), max(head_gw_model)+0.5)
        ax.set_ylim(min(head_ogs[:,0]), max(head_gw_model)+0.5)
        
        ''' Aus confined:
        ax.set_xlim(min(min(head_gw_model, head_ogs_obs_mean)), max(max(head_gw_model, head_ogs_obs_mean)))
        ax.set_ylim(min(min(head_gw_model, head_ogs_obs_mean)), max(max(head_gw_model, head_ogs_obs_mean)))
        '''
        
        ax.scatter(head_gw_model, head_ogs[:, 0])
        ax.plot(head_gw_model, head_gw_model)
        ax.set_title("scatter analytical vs numerical")
        ax.set_xlabel('head analytical [m]')
        ax.set_ylabel('head numerical [m]')     
            
#        plt.show()
        fig.savefig(str(os.path.basename(cwd))+".png")
        return fig
# =============================================================================
# execute script
# =============================================================================
mean_head = np.mean(head_ogs[:,0])
print("Mean groundwater head: " + str(mean_head) + " m")
split(getlist(data_H_gw_model))
rmse = error(head_ogs, head_gw_model)
now = time.time()
print("ogs-data script runtime: " + str(now - then))    
fig = plot()