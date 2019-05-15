#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 01:09:48 2018

@author: houben
"""
import numpy as np
import matplotlib.pyplot as plt

zeit = np.arange(0, 86400 * 365 * 11, 86400)

normal = np.random.lognormal(mean=0.0, sigma=0.1, size=365 * 10)

hist = plt.hist(normal)

mu, sigma = 3.0, 1.0  # mean and standard deviation
s = np.random.lognormal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 100, normed=True, align="mid")


wert = np.sin(zeit)

plt.plot(normal)


count
