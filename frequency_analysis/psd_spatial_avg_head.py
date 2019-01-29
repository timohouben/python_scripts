#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 12:30:33 2019

@author: houben
"""

import os
from fft_psd_head import fft_psd
import numpy as np
import matplotlib.pyplot as plt


# Calculate the PSD for average head time series for multiple folders
cutoff = 0
detrend = True

path_to_multiple_projects = raw_input(
    "Specfy parent directory to multiple model runs: "
)
project_folder_list = [
    f for f in os.listdir(str(path_to_multiple_projects)) if not f.startswith(".")
]
try:
    project_folder_list.remove("fitting_results")
except ValueError:
    pass
try:
    project_folder_list.remove("shh_anal_test")
except ValueError:
    pass
project_folder_list.sort()

results = []
for project_folder in project_folder_list:
    fft_data = np.loadtxt(
        path_to_multiple_projects
        + "/"
        + project_folder
        + "/"
        + "spatial_averaged_head_timeseries.txt"
    )
    recharge = np.loadtxt(
        path_to_multiple_projects + "/" + project_folder + "/" + "rfd_curve#1.txt"
    )

    if cutoff != None:
        print(
                "First "
                + str(cutoff)
                + " data points in time series were deleted due to instationary of the aquifer."
                )
        fft_data = fft_data[cutoff:]
        recharge = recharge[cutoff:]


    results.append(fft_psd(
        fft_data=fft_data,
        recharge=recharge,
        method="scipyffthalf",
        aquifer_thickness=30,
        aquifer_length=1000,
        path_to_project=path_to_multiple_projects + "/" + project_folder,
        obs_point="spatial_averaged_head",
        fit=True,
        time_step_size=86400,
        savefig=True,
        detrend=detrend
    )[:4]
    )
    
Ss = [1.20E-03,
1.10E-03,
1.00E-03,
9.00E-04,
8.00E-04,
7.00E-04,
6.00E-04,
5.00E-04,
4.00E-04,
3.00E-04,
2.00E-04,
1.00E-04,
9.00E-05
]

plt.plot(Ss,[x[2] for x in results], label="Spec. Storage", lw="0.001", marker="o")
plt.plot(Ss,Ss,color="black")
plt.xlabel("input values")
plt.ylabel("derived values")
plt.legend()
plt.savefig(path_to_multiple_projects + "/" + "Ss_spat_averaged_head_" + str(detrend) + "_" + str(cutoff) + ".png")
plt.close()


D = [8.33E-03,
9.09E-03,
1.00E-02,
1.11E-02,
1.25E-02,
1.43E-02,
1.67E-02,
2.00E-02,
2.50E-02,
3.33E-02,
5.00E-02,
1.00E-01,
1.11E-01]

plt.plot(D,[x[3] for x in results], label="Diffusivity", lw="0.001", marker="o")
plt.plot(D,D,color="black")
plt.xlabel("input values")
plt.ylabel("derived values")
plt.legend()
plt.savefig(path_to_multiple_projects + "/" + "D_spat_averaged_head_" + str(detrend) + "_" + str(cutoff) + ".png")
plt.close()
   
K = [1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05,
1.00E-05]

plt.plot(K,[x[1] for x in results], label="hydr. Cond.", lw="0.001", marker="o")
plt.plot(K,K,color="red",lw="0.001", marker="o")
plt.xlabel("input values")
plt.ylabel("derived values")
plt.legend()
plt.savefig(path_to_multiple_projects + "/" + "K_spat_averaged_" + str(detrend) + "_" + str(cutoff) + ".png")
plt.close()
    
    
    
    