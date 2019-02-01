#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:46:09 2019

@author: houben

Script to fit the analytical solution of the power spectrum to the power spectrum of the observed data

"""
import scipy.optimize as optimization
from shh_analytical import shh_analytical
from functools import partial as prt






def shh_analytical_fit(Sww, Shh, frequency, location, length, m, n, norm):
    partial = prt(shh_analytical, x=location, L=length, m=m, n=n, norm=norm)
    initial_guess = [1,1]
    popt, pcov = optimization.curve_fit(
        partial,
        (frequency, Sww),
        Shh,
        p0=initial_guess
    )
    return popt, pcov


if __name__ == "__main__":
    popt, pcov = shh_analytical_fit(power_spectrum_input, power_spectrum_output, frequency_input, location=500, length=1000, m=2, n=2, norm=False)
    plt.loglog(power_spectrum_output, label="data", color="blue")
    plt.loglog(shh_analytical((frequency_input,power_spectrum_input), popt[0], popt[1], x=500, L=1000, norm=False), label="fit", color="red")
    plt.legend()
    plt.show()
