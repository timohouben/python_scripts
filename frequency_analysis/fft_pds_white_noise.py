#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
script to visualize the FFT and power spec density
"""
import scipy.fftpack as scpfft
import numpy as np
import matplotlib.pyplot as plt


#############
# script to visualize the 

########
# generate white noise

mean = 0
std = 3
num_samples=1000
seed=516169845
np.random.seed(seed)
mm_day = np.random.normal(mean, std, size=num_samples)


time_step_size = 0.1

fft = scpfft.fft(mm_day)
freq = scpfft.fftfreq(len(mm_day), time_step_size)    
ind=np.arange(1,len(mm_day)/2+1)
psd=abs(fft[ind])**2+abs(fft[-ind])**2

fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(1,2,1)
ax.plot(mm_day, color='b')
ax.set_title('white noise')
ax.grid(color='grey', linestyle='--', linewidth=0.5)
ax.tick_params(labelbottom='off')

ax = fig.add_subplot(1,2,2)
ax.plot(freq[ind-1],psd[0:])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title('power spectral density')
ax.set_xlabel('frequency')
ax.set_ylabel('intensity')

#################