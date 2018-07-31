#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
script to visualize the FFT and power spec density
"""
import scipy.fftpack as scpfft
import numpy as np
import matplotlib.pyplot as plt


#############
# script to visualize the power spectra


# generate sinus sample data
x = np.arange(0, 50, 0.01)

sin1 = np.sin(x)
sin2 = np.sin(2*x)
sin3 = 0.5 * np.sin(3*x)
sin = sin1 + sin2 + sin3

time_step_size = 0.01

fft = scpfft.fft(sin)
freq = scpfft.fftfreq(len(sin), time_step_size)    
ind=np.arange(1,len(sin)/2+1)
psd=(abs(fft[ind])**2+abs(fft[-ind])**2)#/len(fft)

fig = plt.figure(figsize=(15,8))

ax = fig.add_subplot(4,2,1)
ax.plot(x[:2000],sin[:2000], label='1', color='b')
ax.set_ylabel('signal')
ax.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi, 2.5*np.pi, 3*np.pi, 3.5*np.pi, 4*np.pi, 4.5*np.pi,  5*np.pi, 5.5*np.pi, 6*np.pi])
ax.grid(color='grey', linestyle='--', linewidth=0.5)
ax.tick_params(labelbottom='off')

ax = fig.add_subplot(4,2,3)
ax.plot(x[:2000],sin1[:2000], color='green')
ax.set_ylabel('sin(x)')
ax.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi, 2.5*np.pi, 3*np.pi, 3.5*np.pi, 4*np.pi, 4.5*np.pi,  5*np.pi, 5.5*np.pi, 6*np.pi])
ax.grid(color='grey', linestyle='--', linewidth=0.5)
ax.tick_params(labelbottom='off')

ax = fig.add_subplot(4,2,5)
ax.plot(x[:2000],sin2[:2000], color='orange')
ax.set_ylabel('sin(2x)')
ax.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi, 2.5*np.pi, 3*np.pi, 3.5*np.pi, 4*np.pi, 4.5*np.pi,  5*np.pi, 5.5*np.pi, 6*np.pi])
ax.grid(color='grey', linestyle='--', linewidth=0.5)
ax.tick_params(labelbottom='off')

ax = fig.add_subplot(4,2,7)
ax.plot(x[:2000],sin3[:2000], label='1', color='brown')
ax.set_ylabel('1/2 * sin(3x)')
ax.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi, 2.5*np.pi, 3*np.pi, 3.5*np.pi, 4*np.pi, 4.5*np.pi,  5*np.pi, 5.5*np.pi, 6*np.pi])
ax.set_xticklabels(["$0$", r"", r"$\pi$", r"", r"$2\pi$", r"", r"$3\pi$", r"", r"$4\pi$", r"", r"$5\pi$", r"", r"$6\pi$"])
ax.grid(color='grey', linestyle='--', linewidth=0.5)

ax = fig.add_subplot(1,2,2)
ax.plot(freq[ind],psd)
ax.set_xscale("log")
#ax.set_yscale("log")
ax.set_title('power spectral density')
ax.set_xlabel('frequency')

#################
