#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
# -*- coding: utf-8 -*


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
