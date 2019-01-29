#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:46:09 2019

@author: houben

Script to fit the analytical solution of the power spectrum to the power spectrum of the observed data

"""
import scipy.optimize as optimization
from shh_analytical import shh_analytical

def shh_analytival_fit(Sww, Shh, frequency, x, L):
    x = x
    L = L
    initial_guess = [1,1]
    popt, pcov = optimization.curve_fit(
        shh_analytical,
        (frequency, Sww),
        Shh,
        p0=initial_guess
    )

if __name__ == "__main__":
    plt.loglog(power_spectrum_result, label="data", color="blue")
    plt.loglog(shh_analytical((frequency_input,power_spectrum_input), popt[0], popt[1], norm=True), label="fit", color="red")
    plt.legend()
    plt.show()
