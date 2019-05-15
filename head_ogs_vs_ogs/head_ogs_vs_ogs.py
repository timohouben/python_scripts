#!/usr/bin/env pythonw
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

# =============================================================================
# extract head over x for 1 time step from OGS TECPLOT from vertical observation lines
# =============================================================================

### get list of folders which contain .tec plots
folders = [x for x in os.listdir(".") if os.path.isdir(x)]
folders.sort()

#### variables
#### generate array for head at obs
head_ogs = np.zeros([101, len(folders)])

dir_path = os.path.dirname(os.path.realpath(__file__))


def grabfiles():
    global folder_count
    folder_count = 0
    for file in folders:
        #### grab all files ending with .tec
        tec_files = glob.glob(dir_path + "/" + file + "/" + "*.tec")
        #### sort the list with file names
        tec_files.sort()
        getdata(tec_files)
        print("OGS Model run '" + str(file) + "' done...")
        folder_count = folder_count + 1


# def x_y_data():

# for i in liste:


def getdata(tec_files):
    for k, file_name_ogs in enumerate(tec_files):
        sat_z_data_ogs = []
        data_H_ogs = open(file_name_ogs, "r")
        #### extracts the number of time steps used (inkl. initial condition)
        for i, line in enumerate(data_H_ogs):
            if line[0].isdigit() == True:
                line_numbers = re.findall(
                    "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line
                )
                sat_z_data_ogs.append(line_numbers)
        #### find position of decreasing saturation (first time sat != 1)
        for j, x in enumerate(sat_z_data_ogs):
            try:
                if (
                    sat_z_data_ogs[j][1] > sat_z_data_ogs[j + 1][1]
                ):  # and sat_z_data_ogs[j+1][1] > sat_z_data_ogs[j+2][1] and sat_z_data_ogs[j+2][1] > sat_z_data_ogs[j+3][1]:# and sat_z_data_ogs[j+3][1] > sat_z_data_ogs[j+4][1] and sat_z_data_ogs[j+4][1] > sat_z_data_ogs[j+5][1]:
                    head_ogs[k, folder_count] = sat_z_data_ogs[j][0]
                    break
            except:
                IndexError


"""
ax = plt.subplot(111)
t1 = np.arange(0.0, 1.0, 0.01)
for n in [1, 2, 3, 4]:
    plt.plot(t1, t1**n, label="n=%d"%(n,))

leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)


plt.show()
"""


def plot():
    ax = plt.subplot(111)
    for n, string in enumerate(folders):
        plt.plot(n, n)

    ax.set_title("head over location")
    ax.set_xlabel("location")
    ax.set_ylabel("head [m]")

    plt.show()
    # fig.savefig('plots.png')
    # return fig


grabfiles()
now = time.time()
print("ogs-data script runtime: " + str(now - then))
plot()

"""
#### Defne some useful globar variables
number_obs = sum(1 for line in open("H.OUT", "r")) - 2
obs = np.linspace(0, 1000, 101)
timesteps = 0
#number_rows = 0
file_name_ogs = tec_files[0]

#### extracts the number of time steps used (inkl. initial condition)
data_H_ogs = open(file_name_ogs,'r')
for i, line in enumerate(data_H_ogs):
    if line[0].isdigit() != True:
        timesteps = timesteps + 1
timesteps = timesteps/3



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
# plot stuff
# =============================================================================

def plot():
        #fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(15, 10), subplot_kw={'projection': '3d'})
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle('head analytical vs head numerical')
    
        ax = fig.add_subplot(1, 2, 1)
        ax1 = ax.plot(obs, head_gw_model)
        ax2 = ax.plot(obs, head_ogs[:, 0])
        ax.set_title("head over location")
        ax.set_xlabel('location')
        ax.set_ylabel('head [m]')
        
        ax = fig.add_subplot(1, 2, 2)
        ax3 = ax.scatter(head_gw_model, head_ogs[:, 0])
        ax.set_title("scatter analytical vs numerical")
        ax.set_xlabel('head analytical')
        ax.set_ylabel('head numerical')     
            
        plt.show()
        fig.savefig('plots.png')
        return fig
        
# =============================================================================
# execute script
# =============================================================================
split(getlist(data_H_gw_model))
fig = plot()
"""
