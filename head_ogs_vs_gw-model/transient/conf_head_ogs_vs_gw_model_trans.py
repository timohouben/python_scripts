#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:15:41 2018
@author: houben

This is a script to plot the groundwater head of a confined aquifer from a transient ogs simulation.

Requirements:
- The first CURVE in your .rfd file must be the recharge curve.


TO IMPLE;ENT IN THIS SCRIPT:
    - calculate the baseflow
    - if conditional if only gw head or ogs --> different way to derive the recharge!!!
    - derive time_steps automatically

"""
# =============================================================================
# import modules
# =============================================================================
import matplotlib.pyplot as plt
from ogs5py.reader import readtec_polyline
import numpy as np
import re
import os
from cycler import cycler

# =============================================================================
# global variables set manually
# =============================================================================
which_data_to_plot = 2 # 1: ogs vs gw_model, 2: ogs, 3: gw_model
path_to_project = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/frequency/dupuit_flow/D_18_1000_30_whitenoise_D_18-D_30_homogeneous"
name_of_project_gw_model = "9.00e-04"
name_of_project_ogs = "dupuit_flow1000_30_whitenoise_D_18-D_30_homogeneous"
process = 'GROUNDWATER_FLOW'
which = 'mean'       # min, max, mean
time_steps = 365   # this is the value which is given in the ogs input file .tim. It will result in a total of time_steps+1 times because the initial time is added.
obs_per_plot = ['obs_00100', 'obs_00400', 'obs_00600', 'obs_00800', 'obs_00900']
plt.ioff()

#['obs_0200', 'obs_0400', 'obs_0600', 'obs_0800', 'obs_0950']

# ['obs_0100', 'obs_0200', 'obs_0300', 'obs_0400', 'obs_0500', 'obs_0600', 'obs_0700', 'obs_0800', 'obs_0900', 'obs_0950', 'obs_0990']


#['obs_0100', 'obs_0500', 'obs_0950']

#['obs_0100', 'obs_0200', 'obs_0300', 'obs_0400', 'obs_0500', 'obs_0600', 'obs_0700', 'obs_0800', 'obs_0900', 'obs_0950', 'obs_0990']

#['obs_0100', 'obs_0200', 'obs_0300', 'obs_0400', 'obs_0500', 'obs_0600', 'obs_0700', 'obs_0800', 'obs_0900'] 

# =============================================================================
# format plots
# =============================================================================

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}
plt.rc('font', **font)
if which_data_to_plot == 1:
    plt.rc('axes', prop_cycle=(cycler('color', 
                                  ['b', 'b', 'g', 'g', 'r', 'r', 'c', 'c', 'm', 'm', 'k', 'k', 'y', 'y'])))
else:
    plt.rc('axes', prop_cycle=(cycler('color', 
                                  ['b', 'g', 'r', 'c', 'm', 'k', 'y'])))

# ========================================== ===================================
# global variables set automatically
# =============================================================================

recharge = []

if __name__ == "__main__":
    # read the tec files as dict
    print('Reading .tec-files...')
    tecs = readtec_polyline(task_id=name_of_project_ogs,task_root=path_to_project)
    print('Reading of .tec-files finished.')
    # Save the dict
    print('Saving data...')
    np.save(str(path_to_project) + '/' + 'tecs.npy', tecs)
    print('Saving finished.')

    time_s = tecs[process][obs_per_plot[0]]["TIME"]
    time_d = time_s / 86400

def load_tecs():
    '''
    Load tecs from file.
    '''
    tecs =  np.load(str(path_to_project) + '/' + 'tecs.npy').item()
    return tecs

# =============================================================================
# =============================================================================
# =============================================================================
# GW Model de Rooij 2012
# read timeseries for different observation points from H.OUT of gw-model de Rooij 2012
# =============================================================================
'''    
try:
    n_locations_gw_model = sum(1 for line in open(str(path_to_project) + "/" + str(name_of_project_gw_model) + '/OutputLocations.in' , "r"))
except IOError:
    print("No data for gw model de Rooij, 2012")
'''

def convert_obs_list_to_index(obs):
    '''
    Function to convert the input locations in variable obs_per_plot to indexes 
    to pick the right observation point from the gw_model data.
    '''
    # Lennart: allgemeiner fassen
    obs_index = int(obs[4:])/10
    return obs_index

def getlist_gw_model(path_filename):
    '''
    Obtain a list from the gw_model raw data without header.
    '''
    with open(path_filename, "r") as data_file:
        data_file = open(path_filename, "r")
        raw_data = []
        for string in data_file:
            line = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", string)
            raw_data.append(line)
        del raw_data[0:2]
    data_file.closed
    return raw_data


def split_gw_model(raw_data, index=2):
    '''
    Split the list from getlist_gw_model() into its component. Specify it as 
    argument for the parameter value. Choose between head, time and obs(ervation point).
    The argument index specifies which parameter should be extracted:
        time : 0
        obs  : 1
        head : 2
    '''

    ### Lennart: raw_data_split = raw_data[:][0], als array
    ### Lennart: anstatt head, time obs vielleicht nur index der jeweiligen reihe als argument in der funktion
    
    raw_data_split=[]
    for i in range(0,len(raw_data)):
       raw_data_split.append(float(raw_data[i][index]))
    return raw_data_split
    
def make_array_gw_model(raw_data_split):
    '''
    Make an array from the return of split_gw_model() and safe for each 
    observation point a time series in a column.
    One row contains the data for all observation points at one time step.
    '''
    array_head_gw_model = np.zeros([time_steps+1,n_locations_gw_model])
    i = 0
    k = 0
    for i in range(time_steps+1):
        j = 0
        for j in range(n_locations_gw_model):
            array_head_gw_model[i,j] = raw_data_split[k]
            k += 1
    return array_head_gw_model        

# activate the following lines to save the array in a variable 
# array = make_array_gw_model(
#                    split_gw_model(getlist_gw_model(str(path_to_project) 
#                    + str(name_of_project_gw_model) 
#                    + '/H.OUT'), index=2))

def gethead_gw_model_each_obs(array_head_gw_model, obs, save_heads=True):
    '''
    Extract the demanded time series for given observation point and save it in a list.
    '''
    head_gw_model_timeseries_each_obs = []
    ### Lennart:    head_gw_model_timeseries_each_obs = array_head_gw_model[:,obs]
    
    for line in array_head_gw_model[:,obs]:
        head_gw_model_timeseries_each_obs.append(line)
    np.savetxt(str(path_to_project) + '/' + 'head_gw_model_' + str(obs) + '.txt', head_gw_model_timeseries_each_obs)         
    return head_gw_model_timeseries_each_obs

# =============================================================================
# load .rfd with recharge CURVE or extract recharge from R. in from gw_model
# =============================================================================
def get_curve(path_to_project, name_of_project_ogs, curve=1, mm_d=True, time_steps=None):
        ''' 
        This function extracts the values from the .rfd-file for a given curve 
        number and stores it in two seperate variables. The first curve is denoted
        with 1. curves should be seperated via a blank line otherwise this could
        eventually throw an error.
        ''' 
        rfd_x = []
        rfd_y = []
        length_of_curves = []
        counter = 0
        found = 0
        rfd = open(str(path_to_project) + "/" + str(name_of_project_ogs) + ".rfd",'r')
        for linenumber, line in enumerate(rfd):
            line = line.strip().split()
            if not line:
                line = ' '
            if line and line[0] == '#CURVES' or line[0] == ['#CURVE']:
                found = 1                
                continue
            if found == 1:
                number = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line[0])
                if not number:  
                    length_of_curves.append(counter)                   
                    found = 0
                    counter = 0
                if line != ' ' and line[0] != '#STOP':    
                    rfd_x.append(float((re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line[0]))[0]))
                    rfd_y.append(float((re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line[1]))[0]))
                    counter += 1
           
        curves_begin = []    
        for i in range(0,len(length_of_curves)):
            if i == 0:
                curves_begin.append(0)
            else:
                curves_begin.append(sum(length_of_curves[0:i]))
            
        if mm_d == True:
            recharge = []
            for item in rfd_y:
                recharge.append(item*86400*1000)
            rfd_y = recharge
        
        if time_steps:
            return rfd_x[curves_begin[curve-1]:(curves_begin[curve-1]+time_steps+1)], rfd_y[curves_begin[curve-1]:(curves_begin[curve-1]+time_steps+1)]
        else:
            return rfd_x[curves_begin[curve-1]:(curves_begin[curve-1]+length_of_curves[curve-1])], rfd_y[curves_begin[curve-1]:(curves_begin[curve-1]+length_of_curves[curve-1])]
        
""" alte funktion:        
def getrecharge(path_to_project, name_of_project_ogs, time_steps, mm_d=True):
    ''' 
    This function extracts the recharge from the .rfd-file BUT ONLY FOR THE GIVEN
    NUMBER OF TIMESTEPS in the variable time_steps.
    The curve for the recharge must be the FIRST one in the .rdf-file.
    '''
    
    ### erst innerhalb der funktion nach which_data_to_plot fragen
    ### finall auslagern in eine kleine funktion
    
    recharge = []
    if which_data_to_plot == 1 or which_data_to_plot == 2:
        counter = 0
        rfd = open(str(path_to_project) + "/" + str(name_of_project_ogs) + ".rfd",'r')
        for linenumber, line in enumerate(rfd):
                # line[4] only works when .rfd was saved via the ogs5py api. For other files, select another value
            if line[4].isdigit() == True and counter <= time_steps+1:
                line = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
                recharge.append(line[1])
                counter += 1    
    elif which_data_to_plot == 3:
        recharge_raw = open(str(path_to_project) + "/" + str(name_of_project_gw_model) + "/R.in",'r')
        for line in recharge_raw:
            value = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)
            if value != []:
                recharge.append(value[1])
    
    if mm_d == True:
        recharge_new=[]
        for item in recharge:
            recharge_new.append(float(item)*86400*1000)
        recharge = recharge_new
    return recharge
"""
# =============================================================================
# =============================================================================
# =============================================================================
# OGS
# get the head for maximum, minimum and mean values for given observation point(s)
# =============================================================================
    
def gethead_ogs_each_obs(process, observation_point, which, time_steps, tecs, path_to_project=path_to_project, single_file=False, save_heads=True):
    '''
    Depending for which position on the observations line you want to plot the 
    head, it will return the min, max or mean head. You need to specify this 
    argument when you call the function plot_obs_vs_time.
    '''
    if single_file == False or single_file == None:
        number_of_columns = tecs[process][observation_point]["HEAD"].shape[1]
        if which == 'max':
            # select the maximum value (i.e. the uppermost) of polyline as long as polylines are defined from bottom to top
            head_ogs_timeseries_each_obs = tecs[process][observation_point]["HEAD"][:,number_of_columns-1]
        elif which == 'min':
            # select the minimum value (i.e. the lowermost) of polyline as long as polylines are defined from bottom to top
            head_ogs_timeseries_each_obs = tecs[process][observation_point]["HEAD"][:,0]    
        elif which == 'mean':
            head_ogs_timeseries_each_obs=[]
            for step in range(time_steps+1):
                # calculates the mean of each time step
                head_ogs_timeseries_each_obs.append(np.mean(tecs[process][observation_point]["HEAD"][step,:]))
            head_ogs_timeseries_each_obs = np.asarray(head_ogs_timeseries_each_obs)
        else:
            print('You entered an invalid argument for "which" in function gethead_ogs_each_obs. Please enter min, max or mean.')
            head_ogs_timeseries_each_obs = "nan"
    
    elif single_file == True:
        number_of_columns = tecs["HEAD"].shape[1]
        if which == 'max':
            # select the maximum value (i.e. the uppermost) of polyline as long as polylines are defined from bottom to top
            head_ogs_timeseries_each_obs = tecs["HEAD"][:,number_of_columns-1]
        elif which == 'min':
            # select the minimum value (i.e. the lowermost) of polyline as long as polylines are defined from bottom to top
            head_ogs_timeseries_each_obs = tecs["HEAD"][:,0]    
        elif which == 'mean':
            head_ogs_timeseries_each_obs=[]
            for step in range(time_steps+1):
                # calculates the mean of each time step
                head_ogs_timeseries_each_obs.append(np.mean(tecs["HEAD"][step,:]))
            head_ogs_timeseries_each_obs = np.asarray(head_ogs_timeseries_each_obs)
        else:
            print('You entered an invalid argument for "which" in function gethead_ogs_each_obs. Please enter min, max or mean.')
            head_ogs_timeseries_each_obs = "nan"
    
    # save the head time series as txt for each observation point 
    if save_heads == True:
        np.savetxt(str(path_to_project) + '/' + 'head_ogs_' + str(observation_point) + '_' + str(which) + '.txt', head_ogs_timeseries_each_obs)     
    return head_ogs_timeseries_each_obs

# =============================================================================
# =============================================================================
# =============================================================================
# plot
# =============================================================================

def plot_obs_vs_time(obs_per_plot, which):
    '''
    This function plots the head of ogs versus the time for all given observations 
    points in the variable 'obs_per_plot'. Depending for which position on the
    observations line you want to plot the head, it will return the min, max or
    mean head. You need to specify this argument when you call the function
    plot_obs_vs_time.
    '''
    # check if time_d and recharge have the same shape
    if np.size(time_d) != np.size(recharge):
        print("ERROR: time_d and recharge have different shape!")
        print(str(np.size(time_d)))
        print(str(np.size(recharge)))
    
    # first axis for recharge
    fig, ax1 = plt.subplots(figsize=(14, 10))
    plt.title("head timeseries at different observations points")
    color = 'tab:blue'
    ax1.set_xlabel('time [day]')
    ax1.set_ylabel('recharge [mm/day]', color=color)  # we already handled the x-label with ax1
    ax1.bar(time_d, recharge, width=1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    #ax1.set_ylim(0,0.2)
    #ax1.set_ylim([min(recharge), max(recharge)*2])
    #ax1.set_yticks([0, 1, 2, 3, 4, 5])
   
    # second axis for head
    ax2 = ax1.twinx()
    #ax2.set_ylim(29,35)
    #ax2.set_yticks(np.arange(26,40,0.5))
    color = 'tab:red'

    
    # if conditional to plot ogs vs gw_model, only ogs or only gw model
    # no better quick solution found
    # Lennart: possible solution: write a list containing the arrays for each obs with for loops
    #           and plot the list with a for loop

    
    ###### ogs vs gw_model
    if which_data_to_plot == 1:    
        if which == 'min':
            for obs in obs_per_plot:
                # derive the head for the given observation point from ogs
                head_ogs = gethead_ogs_each_obs(process, obs, 'min', time_steps, tecs)
                ax2.plot(time_d, head_ogs, label = str(obs) + '_' + str(which) + ' OGS',  linestyle='--')
                # derive the head for the given observation point from the gw_model
                head_gw_model = gethead_gw_model_each_obs(make_array_gw_model(
                        split_gw_model(getlist_gw_model(str(path_to_project) + "/"
                        + str(name_of_project_gw_model) 
                        + '/H.OUT'), index=2)), convert_obs_list_to_index(obs))
                ax2.plot(time_d, head_gw_model, label = str(obs) + ' GW_model', linestyle='-')
        elif which == 'max':
            for obs in obs_per_plot:
                head_ogs = gethead_ogs_each_obs(process, obs, 'max', time_steps, tecs)
                ax2.plot(time_d, head_ogs, label = str(obs) + '_' + str(which) + ' OGS',  linestyle='--')
                # derive the head for the given observation point from the gw_model
                head_gw_model = gethead_gw_model_each_obs(make_array_gw_model(
                        split_gw_model(getlist_gw_model(str(path_to_project) + "/" 
                        + str(name_of_project_gw_model) 
                        + '/H.OUT'), index=2)), convert_obs_list_to_index(obs))
                ax2.plot(time_d, head_gw_model, label = str(obs) + ' GW_model', linestyle='-')
        elif which == 'mean':
            for obs in obs_per_plot:
                head_ogs = gethead_ogs_each_obs(process, obs, 'mean', time_steps, tecs)
                ax2.plot(time_d, head_ogs, label = str(obs) + '_' + str(which) + ' OGS',  linestyle='--')
                # derive the head for the given observation point from the gw_model
                head_gw_model = gethead_gw_model_each_obs(make_array_gw_model(
                        split_gw_model(getlist_gw_model(str(path_to_project) + "/" 
                        + str(name_of_project_gw_model) 
                        + '/H.OUT'), index=2)), convert_obs_list_to_index(obs))
                ax2.plot(time_d, head_gw_model, label = str(obs) + ' GW_model', linestyle='-')
        elif which == 'all':
            for obs in obs_per_plot:
                for which in ['mean', 'min', 'max']:
                    head_ogs = gethead_ogs_each_obs(process, obs, which, time_steps, tecs)
                    ax2.plot(time_d, head_ogs, label = str(obs) + '_' + str(which) + ' OGS',  linestyle='--')
                # derive the head for the given observation point from the gw_model
                head_gw_model = gethead_gw_model_each_obs(make_array_gw_model(
                    split_gw_model(getlist_gw_model(str(path_to_project) + "/" 
                    + str(name_of_project_gw_model) 
                    + '/H.OUT'), index=2)), convert_obs_list_to_index(obs))
                ax2.plot(time_d, head_gw_model, label = str(obs) + ' GW_model', linestyle='-')
        else:
            print('You entered an invalid argument for "which" in function plot_obs_vs_time. Please enter min, max, mean or all.')
    
    
    ###### only ogs
    elif which_data_to_plot == 2:
        if which == 'min':
            for obs in obs_per_plot:
                # derive the head for the given observation point from ogs
                head_ogs = gethead_ogs_each_obs(process, obs, 'min', time_steps, tecs)
                ax2.plot(time_d, head_ogs, label = str(obs) + '_' + str(which) + ' OGS',  linestyle='-')
        elif which == 'max':
            for obs in obs_per_plot:               
                head_ogs = gethead_ogs_each_obs(process, obs, 'max', time_steps, tecs)
                ax2.plot(time_d, head_ogs, label = str(obs) + '_' + str(which) + ' OGS',  linestyle='-')
        elif which == 'mean':
            for obs in obs_per_plot:
                head_ogs = gethead_ogs_each_obs(process, obs, 'mean', time_steps, tecs)
                ax2.plot(time_d, head_ogs, label = str(obs) + '_' + str(which) + ' OGS',  linestyle='-')
        elif which == 'all':
            for obs in obs_per_plot:
                for which in ['mean', 'min', 'max']:
                    head_ogs = gethead_ogs_each_obs(process, obs, which, time_steps, tecs)
                    ax2.plot(time_d, head_ogs, label = str(obs) + '_' + str(which) + ' OGS',  linestyle='-')
        else:
            print('You entered an invalid argument for "which" in function plot_obs_vs_time. Please enter min, max, mean or all.')
    
    ###### only gw_model
    elif which_data_to_plot == 3:
        for obs in obs_per_plot:
            # derive the head for the given observation point from the gw_model
            head_gw_model = gethead_gw_model_each_obs(make_array_gw_model(
                            split_gw_model(getlist_gw_model(str(path_to_project) + "/" 
                            + str(name_of_project_gw_model) 
                            + '/H.OUT'), index=2)), convert_obs_list_to_index(obs))
            ax2.plot(time_d, head_gw_model, label = str(obs) + ' GW_model', linestyle='-')
 
      
    ax2.set_ylabel('head [m]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(color='grey', linestyle='--', linewidth=0.5, which='both')
    handles, labels = ax2.get_legend_handles_labels()
    ax1.legend(handles, labels, loc=2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
   
    #maximize the plot window
    #wm = plt.get_current_fig_manager()
    #wm.window.state('zoomed')

    #plt.show()
    # make a string from list obs_per_plot
    obs_per_plot_str = "_".join(obs_per_plot)
    fig.savefig(str(path_to_project) + '/' + str(os.path.basename(str(path_to_project))) + '_' + str(which) + '_' + str(obs_per_plot_str) + '_' + str(which_data_to_plot) + ".png")
    
# =============================================================================
# =============================================================================    
# =============================================================================
# Execute script
# =============================================================================

#gethead_gw_model_each_obs(make_array_gw_model(split_gw_model(getlist_gw_model(str(path_to_project) + str(name_of_project_gw_model) + '/H.OUT'), value='head')), 0)

if __name__ == "__main__":
    rfd_x, recharge = get_curve(path_to_project, name_of_project_ogs, curve=1, mm_d=True, time_steps=time_steps)
    plot_obs_vs_time(obs_per_plot, which=which)