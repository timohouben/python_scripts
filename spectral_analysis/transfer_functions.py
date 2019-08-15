# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division
# ------------------------------------------------------------------------------

def discharge_ftf(f, d, aquifer_length):
    """
    This method computes the discharge frequency tranfer function.
    For further reading see
    Anna Russian et al. 2013:
        Temporal scaling of groundwater discharge in dual and multicontinuum
        catchment models.
        WATER RESOURCES RESEARCH, VOL. 49, 8552–8564,
        doi:10.1002/2013WR014255, 2013

    Parameters
    ----------

    f : 1D-array, float
        The frequency, not angular frequency.
    d : scalar, float
        Aquifer diffusivity.
    aquifer_length : float
        The length of the aquifer from the water divide to the stream.

    """

    import numpy as np

    # calculate angular frequency omega from f
    omega = [i * 2 * np.pi for i in f]

    # calculate the characteristic time
    t_c = aquifer_length**2 * d**(-1)

    def p(w, t_c):
        return np.sqrt(1j*w*t_c)

    theta_q = []
    for w in omega:
        theta_q.append(1 / w / t_c * abs(np.tanh(p(w,t_c)))**2)

    theta_q = np.array(theta_q)
    return theta_q



def discharge_ftf_fit(input, output, time_step_size, aquifer_length, method='scipyffthalf', initial_guess=1e-2):
    """
    This functions computes the power spectrum of input and output with
    power_spectrum from power_spectrum.py. The resulting spectrum is taken to
    derive the aquifer parameters t_c (char time.) and D (T/S) with the
    transfer function of the discharge from Linear Dupuit Aquifer with a
    Dirichlet BC (discharge_ftf).

    Parameters
    ----------

    input : 1D array
        Y-values of time series of input (i.e. recharge)
    output: 1D array
        Y-values of time series of output (i.e. discharge)
    time_step_size : float
        Time in seconds between two time steps / two measurements
    aquifer_length : float
        The length of the aquifer from the water divide to the stream.
    method : string, Default: 'scipyffthalf'
        --> see power_spectrum.py for explanation

    Yields
    ------

    power_spectrum : 1D array
        The power spectrum.
    frequency : 1D array
        The coresponding frequencies.
    t_c : float
        The characteristic time of the aquifer after Russian et al. 2013
    D : float
        The aquifer diffusivity [m^2/s].
    """

    import numpy as np
    from power_spectrum import power_spectrum
    import scipy.optimize as optimization
    from functools import partial as prt

    frequency_input, power_spectrum_result = power_spectrum(input, output, time_step_size, method="scipyffthalf", o_i="oi")

    partial = prt(discharge_ftf, aquifer_length=aquifer_length)
    popt, pcov = optimization.curve_fit(partial, frequency_input, power_spectrum_result, p0=initial_guess)

    theta_q = discharge_ftf(frequency_input, popt[0],1000)

    # check results
    #import matplotlib.pyplot as plt
    #plt.loglog(frequency_input, theta_q)
    #plt.loglog(frequency_input, power_spectrum_result)
    #plt.show()

    return popt, pcov, frequency_input, power_spectrum_result
