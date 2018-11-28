#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division

def calc_aq_param(S, kf, L, b, model, distance=None):
    """
    Function to calculate aquifer parameters a, t, T and D with model input 
    parameters storativity (S), hydraulic conductivity (kf), 
    aquifer length (L) and aquifer thickness (b).
    
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
    model : "linear", "dupuit"
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
    if model == 'linear':
        T = kf * b
        a = ( T * 3. ) / L**2
        t = S / a
        D = L**2. / ( 3. * t )
        Ss = S / b
        returns = [T, kf, Ss, D, a, t] 
        return returns
    
    elif model == 'dupuit':
        if distance == None:
            print('Please specify the distance between your observation point and the river.')
        else:    
            T = kf * b
            a = T / ( b * distance )
            t = ( L**2. * S ) / T
            D = T / S
            Ss = S / b
            returns = [T, kf, Ss, D, a, t] 
            return returns

    else:
        print('Invalid argument: model')