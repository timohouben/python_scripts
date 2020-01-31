# import modules
import sys
import numpy as np
import pandas as pd
# add search path for own modules
sys.path.append("/Users/houben/PhD/python/scripts/spectral_analysis")
# add search path for own modules on eve
sys.path.append("/home/houben/python/scripts/spectral_analysis")
# own modules
from power_spectrum import power_spectrum
from plot_power_spectra import plot_spectrum
from shh_analytical import shh_analytical_fit, shh_analytical
from calc_tc import calc_tc
from processing import detrend
# path to save images
save_path = "/Users/houben/phd/application_spectral_analysis/main/plots/dump"
# Load head
birkach_h = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/birkach_quadratic.txt"
stegaurach_h = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/stegaurach_quadratic.txt"
strullendorf_west_h = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/strull_west_quadratic.txt"
strullendorf_nord_h = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/strull_nord_quadratic.txt"
# load recharge
birkach_r = "/Users/houben/phd/application_spectral_analysis/main/recharge/4416000.0_5522000.0_birkach_recharge.txt"
stegaurach_r = "/Users/houben/phd/application_spectral_analysis/main/recharge/4416000.0_5526000.0_stegaurach_recharge.txt"
strullendorf_west_r = "/Users/houben/phd/application_spectral_analysis/main/recharge/4424000.0_5522000.0_strullendorf_west_recharge.txt"
strullendorf_nord_r = "/Users/houben/phd/application_spectral_analysis/main/recharge/4424000.0_5526000.0_strullendorf_nord_recharge.txt"
names_h = ["birkach", "stegaurach", "strullendorf_west", "strullendorf_nord"]
names_r = ["birkach", "stegaurach", "strullendorf_west", "strullendorf_nord"]
xs = [25800, 600, 7000, 1000]
Ls = [35000, 3000, 8000, 900]
xs = [2300, 2200, 1400, 1100]
Ls = [12000, 2400, 2000, 3000]
xs = [50, 2200, 1400, 1100]
Ls = [100, 2400, 2000, 3000]
paths_h = [birkach_h, stegaurach_h, strullendorf_west_h, strullendorf_nord_h]
paths_r = [birkach_r, stegaurach_r, strullendorf_west_r, strullendorf_nord_r]
#path_r = strullendorf_west_r
#path_h = strullendorf_west_h
#name_h = "strullendorf_west_h"
#name_r = "strullendorf_west_r"
#x = 15000
#L = 20000
m = None
n = None
norm = False
convergence = 0.01
time_step_size = 86400
cut_freq = 1e-4
combine_df = False
detrend_ts = True

Sy = 0.01
T = 0.00001


for name_h, path_h, name_r, path_r, x, L in zip(names_h, paths_h, names_r, paths_r, xs, Ls):
    print("Currently fitting: " + str(name_h))
    # Load the data
    head_df = pd.read_csv(
        path_h,
        sep=" ",
        header=None,
        names=["date", "head"],
    )
    head_df["date"] = pd.to_datetime(head_df["date"])

    # load recharge data
    recharge_df = pd.read_csv(
        path_r,
        sep=" ",
        header=None,
        names=["date", "recharge"],
    )
    recharge_df["date"] = pd.to_datetime(recharge_df["date"])


    if combine_df is True:
        # combine dataframes and remove rows with nans
        combined_df = pd.merge_ordered(recharge_df, head_df, how="inner")
        date_min = combined_df["date"].min()
        date_max = combined_df["date"].max()
        period = combined_df["date"].max() - combined_df["date"].min()
        print("Start/end/length of series where head measurements and recharge overlap: " + str(date_min) + "/" + str(date_max) + "/" + str(period))
        recharge_time_series = combined_df["recharge"].tolist()
        head_time_series = combined_df["head"].tolist()
    else:
        recharge_time_series = recharge_df["recharge"].tolist()
        head_time_series = head_df["head"].tolist()

    if detrend_ts is True:
        head_time_series = detrend(head_time_series)
    else:
        pass

    # convert recharge from mm/d to m/s
    recharge_time_series = [i / 86400 / 1000 for i in recharge_time_series]

    # modify the time series so that both have same length
    # assume: recharge is longest
    recharge_time_series = recharge_time_series[-len(head_time_series):]


    #'''
    ####################################################################
    # artificial data to test the script
    # A) S = 0.5, T = 0.008???, L = 1000, x = 200, white noise
    recharge_time_series = np.loadtxt("/Users/houben/phd/modelling/20190318_spectral_analysis_homogeneous/models_test/1001_24.1127_5.00e-01_8.00e-02/rfd_curve#1.txt")
    head_time_series = np.loadtxt("/Users/houben/phd/modelling/20190318_spectral_analysis_homogeneous/models_test/1001_24.1127_5.00e-01_8.00e-02/head_ogs_obs_00200_mean.txt")
    x = 200
    # B) S = 1.1e-5, T = 1.0 e-3, L = 1000, x = 200, white noise
    recharge_time_series = np.loadtxt("/Users/houben/phd/modelling/20190304_spectral_analysis_homogeneous/models/100_sample2_351_1.10e-05_1.00e-03/rfd_curve#1.txt")
    head_time_series = np.loadtxt("/Users/houben/phd/modelling/20190304_spectral_analysis_homogeneous/models/100_sample2_351_1.10e-05_1.00e-03/head_ogs_obs_00200_mean.txt")
    x = 200
    # C) 1076_border_50_stor_0.0001_rech_mHM, S = 1e-4, T2 = 3e-2
    recharge_time_series = np.loadtxt("/Users/houben/phd/modelling/20190717_SA_hetero_block_2/1076_border_50_stor_0.0001_rech_mHM/rfd_curve#1_y_values.txt")
    head_time_series = np.loadtxt("/Users/houben/phd/modelling/20190717_SA_hetero_block_2/1076_border_50_stor_0.0001_rech_mHM/head_ogs_obs_00500_mean.txt")
    x = 500
    ############
    L = 1000
    time_step_size = 86400
    convergence = 0.01
    m = None
    n = None
    comment = "3_"
    norm = False
    # limits for the spectrum plot
    lims_head = [(1e-9,6e-6),(1e-6,1e12)]
    lims_base = [(1e-9,6e-6),(1e-6,3e1)]
    # cut the data
    # cut_value of frequency
    begin_index = 0
    end_index = 5000
    shift = 5000
    #%matplotlib qt
    import matplotlib.pyplot as plt
    #plt.plot(recharge_time_series)
    #plt.show()
    recharge_time_series = recharge_time_series[begin_index:end_index]
    head_time_series = head_time_series[begin_index+shift:end_index+shift]
    ####################################################################
    #'''

    # calculate the power spectrum: Shh, output to FIT with analy solution only!
    frequency_output, Shh = power_spectrum(
        input=recharge_time_series,
        output=head_time_series,
        time_step_size=time_step_size,
        method="scipyffthalf",
        o_i="o",
    )
    frequency_input, Sww = power_spectrum(
        input=recharge_time_series,
        output=head_time_series,
        time_step_size=time_step_size,
        method="scipyffthalf",
        o_i="i",
    )

    # calculate the fitted power spectra
    Shh_fitted = shh_analytical(
        (frequency_input, Sww),
        Sy,
        T,
        x,
        L,
        m=n,
        n=m,
        norm=norm,
    )


    # get the characteristic time
    tc = calc_tc(L,Sy,T,which="dupuit")
    # define a (discharge constant)
    #a = np.pi ** 2 * T / (4 * L ** 2)
    # define tc (characteristic time scale)
    #tc = Sy / a

    data = np.vstack((Shh, Shh_fitted))

    labels = [
        "Shh numerical",
        "Shh fitted"
    ]
    linestyle = ["-", "-"]
    marker = ["", "d"]
    figtxt ="Derived Parameter:    S = %1.3e, T = %1.3e, tc = %1.3e" % (
        Sy,
        T,
        tc
    )
    print("Currently plotting: " + str(name_h))
    # plot only spectrum of Shh
#    plot_spectrum(
#        [Shh],
#        frequency_output,
#        heading="head spectrum " + name_h,
#        labels=["power_spec_out"],
#        path=save_path,
#        linestyle=["-"],
#        marker=[""],
#        lims=[(2e-9,7e-6),(1e-5,1e7)],
#        name="Shh_" + name_h
#    )

    # plot only spectrum of Sww
    plot_spectrum(
        [Sww],
        frequency_input,
        heading="recharge spectrum " + name_r,
        labels=["power_spec_out"],
        path=save_path,
        linestyle=["-"],
        marker=[""],
        lims=[(2e-9,7e-6),(1e-10,1e4)],
        name="Sww_" + name_r
    )

    # plot Shh and the fitted spectrum
    plot_spectrum(
        data,
        frequency_input,
        heading="Shh and fited spectrum",
        labels=["power_spec_out", "analytical fit"],
        path=save_path,
        linestyle=["-", " "],
        marker=["", "*"],
        figtxt=figtxt,
        #lims=[(2e-9,7e-6),(1e-5,1e7)],
        name="Shh_fitted_" + name_h + "_" + name_r + "_L_" + str(L) + "_x_" + str(x)
    )
    break
