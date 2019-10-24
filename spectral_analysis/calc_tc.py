# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division

# ------------------------------------------------------------------------------


def calc_tc(L, S, T, which="lin"):
    """
    Calculates tc characteristic time scale (Gelhar 1974) in days!

    Parameters
    ----------

    L : float
        Aquifer length [m]
    S : float
        Storativity [-]
    T : float
        Transmissivity [m^2/s]
    which : string
        "linear" : for linear aquifer
        "dupuit" : for dupuit aquifer

    Yields
    ------

    tc : float
        characteristic time scale [day]
    """

    if which == "lin":
        return L ** 2 * S / 3 / T / 86400
    if which == "dupuit":
        from numpy import pi
        return 4 * L ** 2 * S / pi ** 2 / T / 86400
