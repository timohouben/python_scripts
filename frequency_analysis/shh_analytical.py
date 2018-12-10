#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
"""


from __future__ import division
import numpy as np

def shh_analytical(Sww, f, Sy, T, x, L, m=10, n=10):
    """
    Function to analyticaly compute the power spectrum of head with a given
    spectrum of the coresponding recharge process Sww in a phreatic aquifer,
    modeled by a linearized Boussinesq-Equation.
    For further explanations see:
        Liang and Zhang, 2013. Temporal and spatial variation and scaling of 
        groundwater levels in a bounded unconfined aquifer. Journal of 
        Hydrology. http://dx.doi.org/10.1016/j.jhydrol.2012.11.044
        
    
    Definition
    ----------
    def shh_analytical(Sww, omega, Sy, T, x, L, m=1000, n=1000)


    Input
    -----
    Sww         array, power spectrum of recharge as function of frequency 
                omega.
    f           array, frequencies [1/T]
    Sy          float, specific yield [-]
                The specific storage (Ss) in an unconfined aquifer is usually
                much smaller than the specific yield (Sy). Therefore,
                storativity (S[-]) can be approximated with Sy.
                For an unconfined aquifer:
                S = Sy + Ss * b
                with b = saturated thickness?
    T           float, transmissivity [L^2/T]
                T = k * b
                with b = saturated thickness [L] and k = hydr. conductivity 
                [L/T]
    x           float, Location of observed head time series [L]
                x = 0 : dh/dx = 0, x = L : h = h0 (h0 = constant head)
    L           float, aquifer length [L]
    m           integer, number of terms of outer sum, dafault = 1000?
    n           integer, number of terms of inner sum, default = 1000?
            

    Output
    ------
    array, Power spectrum of groundwater head as function of omega


    Restrictions
    ------------
    None.


    References
    ----------
    Liang and Zhang, 2013. Temporal and spatial variation and scaling of 
    groundwater levels in a bounded unconfined aquifer. Journal of 
    Hydrology. http://dx.doi.org/10.1016/j.jhydrol.2012.11.044


    Examples
    --------
    >>> from autostring import astr
    >>> print(astr(esat(293.15),3,pp=True))


    License
    -------
    asd

    """

    # define a (discharge constant)
    a = np.pi ** 2 * T / (4 * L ** 2)
    # define tc (characteristic time scale)
    tc = Sy / a
    # define dimensionless coordinate
    x_dim = x / L
    
    # calculate angular frequency omega from f
    omega = [i*2*np.pi for i in f]
    
    # define two helper function
    def Bm(m, x_dim):
        return np.cos((2 * m + 1) * np.pi * (x_dim / 2)) / (2 * m + 1)

    def Bn(n, x_dim):
        return np.cos((2 * n + 1) * np.pi * (x_dim / 2)) / (2 * n + 1)

    Shh = []
    outer_sum = 0
    print('Omega has length of ' + str(len(omega)))
    for i, freq in enumerate(omega):    
        print('Currently calculating value ' + str(i) + ' of ' + str(len(omega)))
        for j in range(0, m):
            inner_sum = 0
            for k in range(0, n):
                inner_sum += (
                    ((-1) ** (j + k) * Bm(j, x_dim) * Bn(k, x_dim) * Sww[i])
                    / (2 * j ** 2 + 2 * k ** 2 + 2 * j + 2 * k + 1)
                    * (2 * j + 1) ** 2
                    / (((2 * j + 1) ** 4 / tc ** 2) + omega[i] ** 2)
                )
            outer_sum += inner_sum
            print(outer_sum)
        Shh.append(outer_sum * (16 / np.pi ** 2 / Sy ** 2))
    print('Finished')    
    
    # approximation for t >> 1, beta = 2, Shh(omega) prop. omega**2, for more
    # info see Liang and Zhang 2013
    # Shh = [Sww[i]/Sy**2/omega[i] for i in range(0, len(omega))]
    
    Shh = np.asarray(Shh)
    return Shh
