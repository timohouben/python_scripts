#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
sample the parameter space for T and S in an aquifer 
of 1000 m length and a thickness of 30 m to obtain a characteristic time scale 
which is within a given threshold. Modelling period will be 30 years, 10950 days
"""

from calc_tc import calc_t_c
import numpy as np
import matplotlib.pyplot as plt

def rounder(x, min=10):
    '''
    Function to round a float which is smaller than 1 to 2 digits after zeroes.
    '''

    import numpy as np
    # set bins
    for i in range(1,min+1):
        if abs(np.log10(x)) < i:
            return np.around(x, i+1)

# tc is calculated in days

tc_min = 1
tc_max = 10950
scale = 100
Tmin, Tmax = 1e-6, 1e-2
Smin, Smax = 1e-5, 5e-1
length=1000

# distribution should be devided in three segments:
# 1st segment: values greater and close to tc_min
# 2nd segment: values in the middle range
# 3rd segment: values smaller than and close to tc_max

# algorythm: generate random parameters combinations until certain criteria is met

# define the segments, maybe relativ could be better
tc_bin1 = 30
tc_bin2 = 8000

# define counter for each segment
segment1 = 0
segment2 = 0
segment3 = 0

tc = []

binsize = 300

# generate random values
while segment1 < binsize:
    S = np.random.rand()
    T = np.random.rand()
    if S > Smin and S < Smax and T < Tmax and T > Tmin:
        tc_temp = calc_t_c(length, S, T)
        if tc_temp < tc_bin1 and tc_temp > tc_min:
            segment1 += 1
            tc.append([tc_temp,S,T])
    tc_temp = 0

while segment2 < binsize:
    S = np.random.rand()
    T = np.random.rand()
    if S > Smin and S < Smax and T < Tmax and T > Tmin:
        tc_temp = calc_t_c(length, S, T)       
    if tc_temp < tc_bin2 and tc_temp > tc_bin1:
        segment2 += 1
        tc.append([tc_temp,S,T])
    tc_temp = 0    
    
while segment3 < binsize:
    S = np.random.rand()
    T = np.random.rand()
    if S > Smin and S < Smax and T < Tmax and T > Tmin:
        tc_temp = calc_t_c(length, S, T)    
    if tc_temp < tc_max and tc_temp > tc_bin2:     
        segment3 += 1
        tc.append([tc_temp,S,T])
    tc_temp = 0        
    
tc.sort()
tc = np.asarray(tc)
plt.hist(tc[:,0], bins = 10000)
plt.show()

'''
from matplotlib.mlab import frange
tc = []
item_old = 0
ktem_old = 0
for i, item in enumerate(frange(Tmin, Tmax, (Tmax-Tmin)/scale)):
    item = rounder(item)    
    for k, ktem in enumerate(frange(Smin, Smax, (Smax-Smin)/scale)):
        ktem = rounder(ktem)
        print(item)
        tc_temp = calc_t_c(length,ktem,item)
        if tc_temp > tc_min and tc_temp < tc_max:
            tc.append([tc_temp,item,ktem])
tc.sort()
tc = np.asarray(tc)

import matplotlib.pyplot as plt

plt.hist(tc[:,0], bins=10)
plt.show()



            
test = [0.002456445, 0.00045375456, 0.0000012456, 0.00236456, 0.0123456465]

result = []
for t in test:
    result.append(rounder(t))
'''    