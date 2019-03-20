# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division

# ------------------------------------------------------------------------------


def calc_tc(L, S, T):
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

    Yields
    ------

    tc : float
        characteristic time scale [day]
    """
    return L ** 2 * S / 3 / T / 86400
