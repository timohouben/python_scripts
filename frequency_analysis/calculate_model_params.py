#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np


def calc_aq_param(Ss, kf, L, b, model, distance=None, a_alterna=False):
    """
    Function to calculate aquifer parameters T, S, D, a and t with model input 
    parameters storativity (Ss), hydraulic conductivity (kf), 
    aquifer length (L) and aquifer thickness (b).
    If you use the distance for linear model, a will be calculated with respect
    to the location in the aquifer, i.e. distance to the river with the equation
    a = beta * T / L^2, beta^2 = 4 / (1 - (x/L - 1)^4)
    
    Two models can be used:
        Linear Reservoir Model
        Linear Dupuit Model
    
    Gelhar 1974 - Stochastic analysis of Phreatic Aquifers
    Joaquin Jimenez-Martinez et al. - Temporal and spatial scaling of hydraulic
        response to recharge in fractured aquifers: Insights from a frequency
        domain analysis
    
    
    parameters:
    S : int, float
        storativity [L^3 / L^3]
    kf : int, float
        hydraulic conductivity [L/T]
    L : int, float
        aquifer length [L]
    b : int, float
        aquifer thickness [L]
    model : {"linear", "dupuit"}
        which model should be used to calculate parameters
    distance : (optional) int, float
        distance from the river to the observation point
    
    returns:
    a : float 
        discharge constant, recession coefficient [1/T]
    t : float
        characteristic time scale [T]
    T : float
        transmissivity [L^2/T]
    D : float
        diffusivity [L^2/T]
    Ss: float
        specific storage [1/m]    
    
    Linear Reservoir Model:
    a = ( T * 3 ) / L^2
    t = S / a
    T = kf * b
    D = L^2 / ( 3 * t ) = T / S = K / Ss
    
    Linear Dupuit Model:
    a = T / ( b * ( distance ) )
    t = ( L^2 * S ) / T
    T = kf * b
    ? D = L^2 / ( 3 * t ): This formulation ist not valid for Dupuit Aquifers?
    D = T / S = K / Ss
    """
    if model == "linear":
        T = kf * b
        S = Ss * b
        if distance == None:
            a = (T * 3.0) / L ** 2
        else:
            print("Discharge constant a for linear model is calculated with respect to the location in the aquifer. Distance to the river: " + str(distance))
            beta = np.sqrt(4 / (1 - ( distance/L - 1)**4 ))
            a = beta * T / L**2
        t = S / a
        D = T / S
        returns = [T, kf, Ss, D, a, t]
        return returns

    elif model == "dupuit":
        if distance == None:
            print(
                "Please specify the distance between your observation point and the river."
            )
        else:
            T = kf * b
            if a_alterna == True:
                print('Discharge constant a for Dupui model is calculated with alternative formulation beta = pi^2/4, see Gelhar 1974.')
                a = np.pi**2 * T / L**2 / 4
            else:    
                a = T / (b * distance)
            S = Ss * b
            t = (L ** 2.0 * S) / T
            D = T / S
            returns = [T, kf, Ss, D, a, t]
            return returns

    else:
        print("Invalid argument: model")
