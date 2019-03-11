#!/usr/bin/env python2
# -*- coding: utf-8 -*-

def get_ogs_parameters(path):
    from ogs5py import OGS
    ogsmodel = OGS(task_root=path)
    ogsmodel.load_model(task_root=path)
    #dir(ogsmodel.mmp)
    #print(vars(ogsmodel.mmp))
    Ss = ogsmodel.mmp.get_block()['STORAGE'][0][1]
    kf = ogsmodel.mmp.get_block()['PERMEABILITY_TENSOR'][0][1]
    return Ss, kf

if __name__ == "__main__":
    path = '/Users/houben/PhD/modelling/20190304_spectral_analysis_homogeneous/models/1_sample1_8122_9.50e-05_2.00e-01'
    print(get_ogs_parameters(path))
