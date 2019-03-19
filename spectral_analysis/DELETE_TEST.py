#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
# -*- coding: utf-8 -*
import sys
import numpy as np

sys.path.append("/Users/houben/PhD/python/scripts/spectral_analysis")
from shh_analytical import shh_analytical

import preprocessing
from get_obs import get_obs
import os
from power_spectrum import power_spectrum
from get_ogs_parameters import get_ogs_parameters

# some parameters for the analysis
aquifer_length = 1000
aquifer_thickness = 30
which = "mean"

# for i in range(1,10):
recharge_time_series = np.random.rand(10951)
head_time_series = np.random.rand(10951)
frequency, power_spectrum = power_spectrum(
    input=recharge_time_series,
    output=head_time_series,
    time_step_size=time_step_size,
    method="scipyffthalf",
    o_i="oi",
)
print(frequency, power_spectrum)

del power_spectrum
del frequency
# print(recharge_time_series)

# recharge_time_series = np.random.rand(10951)
# head_time_series = np.random.rand(10951)
frequency, power_spectrum = power_spectrum(
    input=recharge_time_series,
    output=head_time_series,
    time_step_size=time_step_size,
    method="scipyffthalf",
    o_i="oi",
)
