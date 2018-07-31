#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
White Noise Recharge Generator

Whith this tool you can generate simple white noise.

num_samples: amount of days you wish to sample
"""


def whitenoise(low=0, high=3, num_samples=365, seed=516169845, show_plot=False, save_file=False):
    '''
    Generates white noise for e.g. source term ogs numerical simulation.
    low = lowest value
    high = highest value
    num_samples = number of samples
    seed = seed
    show_plot = True/False
    save_file = True/False
    '''
    
#   def whitenoise(mean=0, std=3, num_samples=365, seed=516169845):
    which = 'uniform'
#    which = 'random'
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    np.random.seed(seed)
    # rand_value = np.random.random(mean, std, size=num_samples)
    noise = np.random.uniform(low, high, size=num_samples)
    
    
    # convert all negative values to positive value
#    i = 0
#    for values in rand_value:
#        if values < 0:
#            noise[i] = values * (-1)
#        i = i+1    
    
       
    # set a threshold and erase al values which are smaller
#    i=0
#    for values in rand_value:
#        if values < 0.3:
#            print(values)
#            noise[i] = 0
#        i = i+1 
    
    # calculate sum of your values and print it to screen
    print('Total sum of white noise: ' + str(sum(noise)) + " mm")

    # generate time steps bases on num_samples
    time = np.linspace(0, (num_samples-1)*86400, num_samples)
    
    # concatinate time steps and values
    rfd = np.column_stack((time, noise))
    
    if show_plot == True:
        # plot data and hist
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1,2,1)
        plt.plot(noise)
        ax = fig.add_subplot(1,2,2)
        ax.hist(noise)
    
    if save_file == True:
        # safe .rfd file
        np.savetxt(str(which) + '_' + str(low) + '_' + str(high) + '_' + str(num_samples) + '_' + str(seed) + ".rfd",rfd)
 
    return noise

