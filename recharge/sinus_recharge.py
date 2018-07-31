#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Short script to generate sinusoidal recharge. 

"""

# set the number of time steps. You can find this value in the .tim-file from ogs.
number_of_time_steps = 0
# set the time step size in seconds. This value can be set arbitralily.
time_step_size = 0
# set the oscillation duration
osc_duration = 0



def sinusrecharge(number_of_time_steps=365, time_step_size=86400, osc_duration=30, damping=0.004):
    '''
    This function generates a sinusoidal shaped recharge and stores it in a file.
    number_of_time_steps :  specify the number of time steps you want to generate.
                            Time step 0 not included.
    time_step_size       :  Specify the size of one time step in seconds
    osc_duration         :  Specify the duration of one period as integer. 
                            It will be multiplied with your time_step_size.
    damping              :  Specify a damping factor to scale (e.g. 0.005).                      
    '''
    # convert to string to use .isdigit() and ask for integer
    osc_duration_str = str(osc_duration)
    
    if osc_duration <= 1 or osc_duration_str.isdigit() == False:
        print('The oscillation duration must be set to a value greater than 2 and it needs to be an integer.')
    else:
        import numpy as np
        import matplotlib.pyplot as plt
        import cmath as math
        import time as tim
        timestr = tim.strftime("%Y%m%d-%H%M%S")
        
        # generate time for .rfd-file
        time = np.linspace(0, (number_of_time_steps)*time_step_size, number_of_time_steps+1)
    
        sin = []
        for value in time:
            line = math.sin(value/(osc_duration*time_step_size)*math.pi) * damping
            sin.append(float(line.real))
        
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(1,1,1)
        ax.plot(time,sin)
        
        # convert to mm/second
        mm_s = np.asarray(sin)/86400
        
        # concatinate time steps and values
        rfd = np.column_stack((time, mm_s))
        
        # safe .rfd file
        np.savetxt(str(number_of_time_steps) + '_' + str(time_step_size) + '_' + str(osc_duration) + '_' + str(damping) + ".rfd",rfd)
        
        # calculate the yearly averaged recharge
        mm_y = sum(mm_s * time_step_size * number_of_time_steps * (365*86400/(time_step_size * number_of_time_steps)))
        mm_y_test = sum(mm_s * time_step_size * number_of_time_steps)

        print(mm_y_test)
        print('The averaged yearly recharge is ' + str(mm_y) + ' mm.')