#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:56:28 2019

@author: houben
"""

# calculate t_c
# t_c = L^2 * S / 3 / T


def calc_t_c(L, S, T):
    return L ** 2 * S / 3 / T / 86400
