# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division
# ------------------------------------------------------------------------------

def get_ogs_parameters(path):
    from ogs5py import OGS

    ogsmodel = OGS(task_root=path)
    ogsmodel.load_model(task_root=path)
    # dir(ogsmodel.mmp)
    # print(vars(ogsmodel.mmp))
    # for any reason is this configuration not working with python2
    # Ss = ogsmodel.mmp.get_block()['STORAGE'][0][1]
    # kf = ogsmodel.mmp.get_block()['PERMEABILITY_TENSOR'][0][1]
    # print(vars(ogsmodel.mmp))

    # this configuration is probably only working for this kind of OGS setups
    # because it is indexing the contents of the mmp file and not refering to a
    # dictionary with keys!!!
    Ss = float(ogsmodel.mmp.cont[0][1][0][1])
    kf = float(ogsmodel.mmp.cont[0][2][0][1])
    time_step_size = float(ogsmodel.tim.cont[0][3][0][1])
    time_steps = float(ogsmodel.tim.cont[0][3][0][0])
    return Ss, kf, time_step_size, time_steps


if __name__ == "__main__":
    path = "/Users/houben/PhD/modelling/20190304_spectral_analysis_homogeneous/models/100_sample2_351_1.10e-05_1.00e-03"
    print(get_ogs_parameters(path))
