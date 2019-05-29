# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division

# ------------------------------------------------------------------------------


def shh_analytical_man(X, Sy, T, x, L, m=7, n=4, norm=False):
    """
    Function to analyticaly compute the power spectrum of head with a given
    spectrum of the coresponding recharge process Sww in a phreatic aquifer,
    modeled by a linearized Boussinesq-Equation.
    For further explanations see:
        Liang and Zhang, 2013. Temporal and spatial variation and scaling of
        groundwater levels in a bounded unconfined aquifer. Journal of
        Hydrology. http://dx.doi.org/10.1016/j.jhydrol.2012.11.044


    Parameters
    ----------
    X : tuple (f, Sww)
        f : 1D array
            frequencies [1/T], will be internally converted to angular
            frequency omega
        Sww : 1D array
            power spectrum of recharge as function of frequency omega.
    Sy : float
        specific yield [-]
        The specific storage (Ss) in an unconfined aquifer is usually much
        smaller than the specific yield (Sy). Therefore, storativity (S[-]) can
        be approximated with Sy. For an unconfined aquifer:
        S = Sy + Ss * b
        with b = saturated thickness?
    T : float
        transmissivity [L^2/T]
        T = k * b
        with b = saturated thickness [L] and k = hydr. conductivity [L/T]
        Is used to calculate the discharge parameter a.
    x : float
        Location of observed head time series [L]
        x = 0 : dh/dx = 0, x = L : h = h0 (h0 = constant head)
    L : float
        aquifer length [L] from water divide to point of discharge (i.e. stream)
    m : integer
        number of terms of outer sum, dafault = 7
    n : integer
        number of terms of inner sum, default = 4
    norm : bool
        normalize the output spectrum Shh by the input spectrum Sww


    Yields
    ------
    array
        Power spectrum of groundwater head as function of omega


    References
    ----------
    Liang and Zhang, 2013. Temporal and spatial variation and scaling of
        groundwater levels in a bounded unconfined aquifer. Journal of
        Hydrology. http://dx.doi.org/10.1016/j.jhydrol.2012.11.044


    Examples
    --------

    """

    import numpy as np

    f, Sww = X

    # define a (discharge constant)
    a = np.pi ** 2 * T / (4 * L ** 2)
    # define tc (characteristic time scale)
    tc = Sy / a

    # check if distance to river is 0
    if x == L:
        return [np.nan for i in Sww]

    # define dimensionless coordinate
    x_dim = x / L

    # calculate angular frequency omega from f
    omega = [i * 2 * np.pi for i in f]

    # define two helper functions
    def Bm(m, x_dim):
        return np.cos((2 * m + 1) * np.pi * x_dim / 2) / (2 * m + 1)

    def Bn(n, x_dim):
        return np.cos((2 * n + 1) * np.pi * x_dim / 2) / (2 * n + 1)

    Shh = []
    for i, freq in enumerate(omega):
        outer_sum = 0
        # print("Currently calculating value " + str(i) + " of " + str(len(omega)))
        for j in range(0, m):
            inner_sum = 0
            for k in range(0, n):
                inner_sum += (
                    ((-1) ** (j + k) * Bm(j, x_dim) * Bn(k, x_dim) * Sww[i])
                    / (2 * j ** 2 + 2 * k ** 2 + 2 * j + 2 * k + 1)
                    * (
                        (2 * j + 1) ** 2
                        / (((2 * j + 1) ** 4 / tc ** 2) + omega[i] ** 2)
                    )
                )
            outer_sum += inner_sum
        Shh.append(outer_sum * (16 / np.pi ** 2 / Sy ** 2))

    # approximation for t >> 1, beta = 2, Shh(omega) prop. omega**2, for more
    # info see Liang and Zhang 2013
    # Shh = [Sww[i]/Sy**2/omega[i] for i in range(0, len(omega))]

    if norm == True:
        Shh_Sww = [value / Sww[i] for i, value in enumerate(Shh)]
        Shh_Sww = np.asarray(Shh_Sww)
        return Shh_Sww
    else:
        Shh = np.asarray(Shh)
        return Shh


def shh_analytical(X, Sy, T, x, L, m=None, n=None, norm=False, convergence=0.01):
    """
    Function to analyticaly compute the power spectrum of head with a given
    spectrum of the coresponding recharge process Sww in a phreatic aquifer,
    modeled by a linearized Boussinesq-Equation.
    For further explanations see:
        Liang and Zhang, 2013. Temporal and spatial variation and scaling of
        groundwater levels in a bounded unconfined aquifer. Journal of
        Hydrology. http://dx.doi.org/10.1016/j.jhydrol.2012.11.044

    In contrast to shh_analytical_man, where you have to specifiy the number of
    iterations manuallay, here you have to provide a convergence criterion.


    Parameters
    ----------
    X : tuple (f, Sww)
        f : 1D array
            frequencies [1/T], will be internally converted to angular
            frequency omega
        Sww : 1D array
            power spectrum of recharge as function of frequency omega.
    Sy : float
        specific yield [-]
        The specific storage (Ss) in an unconfined aquifer is usually much
        smaller than the specific yield (Sy). Therefore, storativity (S[-]) can
        be approximated with Sy. For an unconfined aquifer:
        S = Sy + Ss * b
        with b = saturated thickness?
    T : float
        transmissivity [L^2/T]
        T = k * b
        with b = saturated thickness [L] and k = hydr. conductivity [L/T]
        Is used to calculate the discharge parameter a.
    x : float
        Location of observed head time series [L]
        x = 0 : dh/dx = 0, x = L : h = h0 (h0 = constant head)
    L : float
        aquifer length [L] from water divide to point of discharge (i.e. stream)
    m : integer
        number of terms of outer sum, dafault = None
        In this function, m is actually doing nothing. Is is still there to ensure
        compatibility with the function to fit (shh_analytical_fit) which should
        be taken for both, shh_analytical and shh_analytical_man.
    n : integer
        number of terms of inner sum, default = None
        In this function, m is actually doing nothing. Is is still there to ensure
        compatibility with the function to fit (shh_analytical_fit) which should
        be taken for both, shh_analytical and shh_analytical_man.
    norm : bool
        normalize the output spectrum Shh by the input spectrum Sww
    convergence : float
        Convergence criterion. If new iteration of series adds less than this
        %-value the series is truncated.


    Yields
    ------
    array
        Power spectrum of groundwater head as function of omega


    References
    ----------
    Liang and Zhang, 2013. Temporal and spatial variation and scaling of
        groundwater levels in a bounded unconfined aquifer. Journal of
        Hydrology. http://dx.doi.org/10.1016/j.jhydrol.2012.11.044


    Examples
    --------

    """

    import numpy as np

    f, Sww = X

    # define a (discharge constant)
    a = np.pi ** 2 * T / (4 * L ** 2)
    # define tc (characteristic time scale)
    tc = Sy / a

    # check if distance to river is 0
    if x == L:
        return [np.nan for i in Sww]

    # define dimensionless coordinate
    x_dim = x / L

    # calculate angular frequency omega from f
    omega = [i * 2 * np.pi for i in f]

    # define two helper functions
    def Bm(m, x_dim):
        return np.cos((2 * m + 1) * np.pi * x_dim / 2) / (2 * m + 1)

    def Bn(n, x_dim):
        return np.cos((2 * n + 1) * np.pi * x_dim / 2) / (2 * n + 1)

    # Count the number of iteration
    # counter_inner = []
    # counter_outer = []

    Shh = []
    for i, freq in enumerate(omega):
        # set outer_sum and single_inner_sum = 1 to pass the first while condition
        outer_sum = 1
        single_outer_sum = 1
        j = 0
        while (single_outer_sum / outer_sum) > convergence:
            # set inner_sum and single_inner_sum = 1 to pass the first while condition
            inner_sum = 1
            single_inner_sum = 1
            k = 0
            while (single_inner_sum / inner_sum) > convergence:
                single_inner_sum = (
                    ((-1) ** (j + k) * Bm(j, x_dim) * Bn(k, x_dim) * Sww[i])
                    / (2 * j ** 2 + 2 * k ** 2 + 2 * j + 2 * k + 1)
                    * (
                        (2 * j + 1) ** 2
                        / (((2 * j + 1) ** 4 / tc ** 2) + omega[i] ** 2)
                    )
                )
                # set inner_sum = single_inner_sum in first iteration
                if k == 0:
                    inner_sum = single_inner_sum
                # increase inner_sum by single_inner_sum for all further iteration
                else:
                    inner_sum += single_inner_sum
                k += 1
            # counter_inner.append(k)
            # print("Needed " + str(k) + " iterations for " + str(j) + ". outer sum.")
            # the result from inner_sum is equal to single_outer_sum. This step ist redundand.
            single_outer_sum = inner_sum
            # set outer_sum = single_outer_sum in first iteration
            if j == 0:
                outer_sum = inner_sum
            # increase outer_sum by single_outer_sum for all further iteration
            else:
                outer_sum += inner_sum
            j += 1
        # counter_outer.append(j)
        # print("Needed " + str(j) + " iterations for " + str(i) + ". value of Sww.")
        Shh.append(outer_sum * (16 / np.pi ** 2 / Sy ** 2))

    # approximation for t >> 1, beta = 2, Shh(omega) prop. omega**2, for more
    # info see Liang and Zhang 2013
    # Shh = [Sww[i]/Sy**2/omega[i] for i in range(0, len(omega))]

    # Show how many iterations where needed to meet the criterion
    # print(counter_inner)
    # print(counter_inner)

    if norm == True:
        Shh_Sww = [value / Sww[i] for i, value in enumerate(Shh)]
        Shh_Sww = np.asarray(Shh_Sww)
        return Shh_Sww
    else:
        Shh = np.asarray(Shh)
        return Shh


def shh_analytical_2015(
    f, Sy, T, x, L, SWW=None, SQQ=None, SHH=None, m=None, n=None, norm=False, convergence=0.01
):
    """
    Function to analytically compute the power spectrum of head with a given
    spectrum of the coresponding recharge process Sww AND a spectrum of a
    fluctuating river SHH AND a flux to the left boundary SQQ in a phreatic
    aquifer, modeled by a linearized Boussinesq-Equation. Default values for SWW,
    SQQ and SHH are "none". If data is provided the corresponding term will be
    switched on and power spectrum of head (Shh) will be calculated.

    For further explanations see:
        Liang and Zhang, 2015. Analyses of uncertainties and scaling of
        groundwater level fluctuations. Hydrology an Earth System Siences.
        doi:10.5194/hess-19-2971-2015

    Parameters
    ----------

    f : 1D array
        frequencies [1/T], will be internally converted to angular
        frequency omega
    SWW : 1D array, default: None
        power spectrum of recharge as functin of frequency omega
    SQQ : 1D array, default: None
        power spectrum of base flow as functin of frequency omega
    SHH : 1D array, default: None
        power spectrum of river height as functin of frequency omega
    Sy : float
        specific yield [-]
        The specific storage (Ss) in an unconfined aquifer is usually much
        smaller than the specific yield (Sy). Therefore, storativity (S[-]) can
        be approximated with Sy. For an unconfined aquifer:
        S = Sy + Ss * b
        with b = saturated thickness?
    T : float
        transmissivity [L^2/T]
        T = k * b
        with b = saturated thickness [L] and k = hydr. conductivity [L/T]
        Is used to calculate the discharge parameter a.
    x : float
        Location of observed head time series [L]
        x = 0 : dh/dx = 0, x = L : h = h0 (h0 = constant head)
    L : float
        aquifer length [L] from water divide to point of discharge (i.e. stream)
    m : integer
        number of terms of outer sum, dafault = None
        In this function, m is actually doing nothing. Is is still there to ensure
        compatibility with the function to fit (shh_analytical_fit) which should
        be taken for both, shh_analytical and shh_analytical_man.
    n : integer
        number of terms of inner sum, default = None
        In this function, m is actually doing nothing. Is is still there to ensure
        compatibility with the function to fit (shh_analytical_fit) which should
        be taken for both, shh_analytical and shh_analytical_man.
    norm : bool
        normalize the output spectrum Shh by the input spectrum Sww
    convergence : float
        Convergence criterion. If new iteration of series adds less than this
        %-value the series is truncated.

    Yields
    ------
    1D array
        Power spectrum of groundwater head as function of omega SWW, SQQ, SHH


    References
    ----------
    Liang and Zhang, 2015. Analyses of uncertainties and scaling of
        groundwater level fluctuations. Hydrology an Earth System Siences.
        doi:10.5194/hess-19-2971-2015

    """

    import numpy as np

    #if type(SWW) == None:
    #    SWW = [0 for i in f]
    #if type(SQQ) == None:
    #    SQQ = [0 for i in f]
    #if type(SHH) == None:
    #    SHH = [0 for i in f]

    print(SHH,SWW,SQQ)

    # define beta
    beta = T / Sy
    # define tc (characteristic time scale)
    tc = Sy * L ** 2 / T
    # check if distance to river is 0
    if x == L:
        return [np.nan for i in Sww]
    # define dimensionless coordinate
    x_dim = x / L
    # calculate angular frequency omega from f
    omega = [i * 2 * np.pi for i in f]

    # define two helper functions
    def bm(m):
        return (2 * m + 1) * np.pi / 2

    def bn(n):
        return (2 * n + 1) * np.pi / 2

    # Count the number of iteration
    # counter_inner = []
    # counter_outer = []

    Shh = []
    for i, freq in enumerate(omega):
        # set outer_sum and single_inner_sum = 1 to pass the first while condition
        outer_sum = 1
        single_outer_sum = 1
        j = 0
        while abs(single_outer_sum / outer_sum) > convergence:
            # set inner_sum and single_inner_sum = 1 to pass the first while condition
            inner_sum = 1
            single_inner_sum = 1
            k = 0
            while abs(single_inner_sum / inner_sum) > convergence:
                single_inner_sum = (
                    (
                        8
                        * beta
                        * bm(j) ** 2
                        * np.cos(bm(j) * x_dim)
                        * np.cos(bn(k) * x_dim)
                    )
                    / (
                        (bn(k) ** 2 + bm(j) ** 2)
                        * (beta ** 2 * bm(j) ** 2 / L ** 4 + omega[i] ** 2)
                    )
                    * (
                        ((-1) ** (k + j) * SWW[i] * L ** 2) / (T ** 2 * bm(j) * bn(k))
                        + SQQ[i] / T
                        + ((-1) ** (k + j) * bm(j) * bn(k) * SHH[i]) / L ** 2
                    )
                )
                # set inner_sum = single_inner_sum in first iteration
                if k == 0:
                    inner_sum = single_inner_sum
                # increase inner_sum by single_inner_sum for all further iteration
                else:
                    inner_sum += single_inner_sum
                k += 1
            # counter_inner.append(k)
            # print("Needed " + str(k) + " iterations for " + str(j) + ". outer sum.")
            # the result from inner_sum is equal to single_outer_sum. This step ist redundand.
            single_outer_sum = inner_sum
            # set outer_sum = single_outer_sum in first iteration
            if j == 0:
                outer_sum = inner_sum
            # increase outer_sum by single_outer_sum for all further iteration
            else:
                outer_sum += inner_sum
            j += 1
        # counter_outer.append(j)
        # print("Needed " + str(j) + " iterations for " + str(i) + ". value of Sww.")
        Shh.append(outer_sum * (16 / np.pi ** 2 / Sy ** 2))

    # approximation for t >> 1, beta = 2, Shh(omega) prop. omega**2, for more
    # info see Liang and Zhang 2013
    # Shh = [Sww[i]/Sy**2/omega[i] for i in range(0, len(omega))]

    # Show how many iterations where needed to meet the criterion
    # print(counter_inner)
    # print(counter_inner)

    if norm == True:
        Shh_Sww = [value / Sww[i] for i, value in enumerate(Shh)]
        Shh_Sww = np.asarray(Shh_Sww)
        return Shh_Sww
    else:
        Shh = np.asarray(Shh)
        return Shh


def shh_analytical_fit(Sww, Shh, f, x, L, m, n, norm):
    """
    Function which should be used to fit the power spectrum to a given
    experimental data set (Shh). Since the shh_anlytical takes additional parameters
    (location, m, n, norm) beside the ones to optimize (Sy, T) the optimization
    needs to use "partial" from functools.

    Parameters
    ----------
    Sww : 1D array
        power spectrum of recharge as function of frequency omega.
    Shh : 1D array
        power spectrum of head as function of frequency omega.
    f : 1D array
        frequencies [1/T], will be internally converted to angular
        frequency omega
    S = Sy + Ss * b
        with b = saturated thickness?
    x : float
        Location of observed head time series [L]
        x = 0 : dh/dx = 0, x = L : h = h0 (h0 = constant head)
    L : float
        aquifer length [L] from water divide to point of discharge (i.e. stream)
    m : integer
        number of terms of outer sum, dafault
    n : integer
        number of terms of inner sum, default
    norm : bool
        normalize the output spectrum Shh by the input spectrum Sww


    Yields
    ------
    popt[0] : float
        specific yield [-]
        The specific storage (Ss) in an unconfined aquifer is usually much
        smaller than the specific yield (Sy). Therefore, storativity (S[-]) can
        be approximated with Sy. For an unconfined aquifer:
        S = Sy + Ss * b
        with b = saturated thickness?
    popt[1] : float
        transmissivity [L^2/T]
        T = k * b
        with b = saturated thickness [L] and k = hydr. conductivity [L/T]
        Is used to calculate the discharge parameter a.
    pcov : array
        covariance matrix
    """

    import scipy.optimize as optimization
    from functools import partial as prt

    partial = prt(shh_analytical, x=x, L=L, m=m, n=n, norm=norm)
    initial_guess = [1e-3, 1e-3]
    bounds = (1e-7, 1)
    popt, pcov = optimization.curve_fit(
        partial, (f, Sww), Shh, p0=initial_guess, method="lm"
    )  # , bounds=bounds)
    return popt, pcov

'''
if __name__ == "__main__":
    popt, pcov = shh_analytical_fit(
        power_spectrum_input,
        power_spectrum_output,
        frequency_input,
        location=500,
        length=1000,
        m=2,
        n=2,
        norm=False,
    )
    plt.loglog(power_spectrum_output, label="data", color="blue")
    plt.loglog(
        shh_analytical(
            (frequency_input, power_spectrum_input),
            popt[0],
            popt[1],
            x=500,
            L=1000,
            norm=False,
        ),
        label="fit",
        color="red",
    )
    plt.legend()
    plt.show()
'''
