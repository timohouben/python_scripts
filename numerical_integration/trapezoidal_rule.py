#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script can be used to numerically calculate the integral with the trapezoidal rule. 
"""

import numpy as np
import glob


# generate example data
# samples = 100
# x_sample = np.sort(np.random.rand(samples)*np.pi*2)
# y_sample = np.array([np.sin(x) for x in x_sample])


filename = glob.glob('*.txt')
if filename == [] or len(filename) > 1:
    print('No files or more than one file in the directory!')
    x_data = []
    y_data = []
else:
    data = np.loadtxt(filename[0])
    x_data = data[:,0]
    y_data = data[:,1]
       
    # identify negative values and how much
    negatives = 0
    for line in y_data:
        if line < 0:
            negatives += 1
          
    # remove lines if negatives values are available
    if negatives > 0:
        negatives = negatives - 1
        # delete negative values at saturated water content
        y_data = np.delete(y_data, [0,negatives])
        x_data = np.delete(x_data, [0,negatives])

    
def trapezoidalrule(x_data=x_data, y_data=y_data):
    '''Function to numerically calculate the integral of a curve
        within a given x range (from min x-value to max x-value given).
        The script will write a file with the input data and the 
        resulting integral in the folder of the script.
        x_data  :   the x coordinates
        y_data  :   f(x)
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import time as tim
    timestr = tim.strftime("%Y%m%d-%H%M%S")
    
    # integral for each interval
    integral_k = np.zeros(len(x_data)-1)
    
    # Trapezioidal rule from https://en.wikipedia.org/wiki/Trapezoidal_rule
    for k in range(1,len(x_data)):
        # interval
        delta_x_k = x_data[k] - x_data[k-1]
        # formula
        integral_k[k-1] = (y_data[k-1] + y_data[k])/2 * delta_x_k
    
    # sum of all integrals
    integral = abs(sum(integral_k))
    print(integral)
    
    # plot the sample data
    plt.plot(x_data, y_data)
    #plt.yscale('log')
    # plot bars as boundaries of intervals
    #plt.bar(x_data, y_data, width=0.01)
    plt.show()

    # save a txt file with the data and the resulting integral
    np.savetxt(str(timestr) + ".txt",np.column_stack((x_data, y_data)))
    
    # open file again and append the integral
    with open(str(timestr) + ".txt", 'a') as file:
        file.writelines('\nThe integral is: ' + str(integral))
    file.close()
