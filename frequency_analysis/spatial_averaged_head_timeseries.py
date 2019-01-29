#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:05:26 2018

- have maximum and minimum observation point in list of observationsS!
"""

import numpy as np
from scipy import integrate
import sys
sys.path.append("/Users/houben/PhD/python/scripts/frequency_analysis/")
from get_obs import get_obs
import os


def spatial_averaged_head_timeseries(path_to_multiple_projects, remove_value=None):
    project_folder_list = [
        f for f in os.listdir(str(path_to_multiple_projects)) if not f.startswith(".")]
    project_folder_list.sort()
    try:
        project_folder_list.remove("fitting_results")
    except ValueError:
        pass    
    
    file_names, obs_names, obs_locs = get_obs(path_to_multiple_projects + '/' + project_folder_list[0]) 
    
    file_names_txt = ['head_ogs_' + obs_names[i] + '_mean.txt' for i, item in enumerate(obs_names)]
    
    files = []
    
    for i, item in enumerate(project_folder_list):
        print('loading ' + item)
        head_timeseries = []    
        for j, jtem in enumerate(file_names_txt):
            file_head = np.loadtxt(path_to_multiple_projects + '/' + item + '/' + jtem)
            head_timeseries.append(file_head)
        files.append(head_timeseries)
    
    for i in range(0,len(project_folder_list)):
        print('Calculating averaged head for ' + project_folder_list[i])
        av_head_timeseries = []
        for k in range(0,len(files[i][j])):
            head_loc = []
            for j in range(0,len(file_names_txt)):
                head_loc.append(files[i][j][k])
            av_head_timeseries.append(integrate.simps(head_loc,obs_locs) / max(obs_locs))
        if remove_value != None:
            av_head_timeseries.pop(remove_value)
        np.savetxt(path_to_multiple_projects + '/' + project_folder_list[i] + '/' + 'spatial_averaged_head_timeseries.txt', av_head_timeseries)
        print("spatial_averaged_head_timeseries.txt has been saved")
    
if __name__ == "__main__":
    path_to_multiple_projects = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30_whitenoise/shh_anal_test"
    # give the index of a value which should be romoved from the head time series. 0 is first index!
    remove_value=None
    spatial_averaged_head_timeseries(path_to_multiple_projects, remove_value)