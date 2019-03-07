#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:21:48 2018

To Improve:
    - not functioning for gw_head!!!
"""

# =============================================================================
# import modules
# =============================================================================

# import sys and set path to the module
import sys

sys.path.append("/Users/houben/PhD/python/scripts/head_ogs_vs_gw-model/transient")
from conf_head_ogs_vs_gw_model_trans import (
    gethead_ogs_each_obs,
    get_curve,
    gethead_gw_model_each_obs,
    make_array_gw_model,
    split_gw_model,
    getlist_gw_model,
    convert_obs_list_to_index,
)
#from ogs5py.reader import readtec_polyline
import scipy.fftpack as fftpack
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#plt.ioff()
import datetime
import os
import scipy.optimize as optimization
import textwrap as tw
from running_mean import moving_average
from calculate_model_params import calc_aq_param
from shh_analytical import shh_analytical


def get_fft_data_from_simulation(
    path_to_project="/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30/testing2/Groundwater@UFZ_eve_HOMO_276_D_4_results",
    single_file="/Users/houben/Desktop/Shh_test_Groundwater@UFZ_eve_HOMO_276_D_30/content/transect_01_ply_obs_0000_t1_GROUNDWATER_FLOW.tec",
    which_data_to_plot=2,
    name_of_project_gw_model="",
    name_of_project_ogs="transect_01",
    process="GROUNDWATER_FLOW",
    which="mean",
    time_steps=8401,
    obs_point="NA",
):
    """
    which_data_to_plot = 1 # 1: ogs, 2: gw_model 3: recharge
    which = 'max'       # min, max, mean
    time_steps = 8400    # this is the value which is given in the ogs input file .tim. It will result in a total of 101 times because the initial time is added.
    methods = ['scipyfftnormt', 'scipyfftnormn', 'scipyfft', 'scipywelch',
               'pyplotwelch', 'scipyperio', 'spectrumperio']
    """

    # =============================================================================
    # initialize the file for output
    # =============================================================================
    with open(str(path_to_project) + "/PSD_output.txt", "a") as file:
        file.write(
            "date time method T_l[m2/s] kf_l[m/s] Ss_l[1/m] D_l[m2/s] a_l t_l[s] T_d[m2/s] kf_d[m/s] Ss_d[1/m] D_d[m2/s] a_d t_d[s] path_to_project observation_point\n"
        )
    file.close()

    # =============================================================================
    # global variables set automatically
    # =============================================================================
    rfd_x, recharge = get_curve(
        path_to_project=path_to_project,
        name_of_project_ogs=name_of_project_ogs,
        curve=1,
        mm_d=False,
        time_steps=time_steps,
    )
    try:
        fft_data = np.loadtxt(
            str(path_to_project)
            + "/"
            + "head_ogs_"
            + str(obs_point)
            + "_"
            + str(which)
            + ".txt"
        )
        print(
            "Loaded heads from file: "
            + str("head_ogs_" + str(obs_point) + "_" + str(which) + ".txt")
        )
    except IOError:
        print("NOT Reading .tec-files, because VTK is not working on EVE...")
        print("Reading .tec-files...")
        print(single_file[-40:])
        #tecs = readtec_polyline(
        #    task_id=name_of_project_ogs,
        #    task_root=path_to_project,
        #    single_file=single_file,
        #)
        print("Finished reading.")

        # =============================================================================
        # get data dependent on which_data_to_plot
        # =============================================================================
        if which_data_to_plot == 2:
            fft_data = gethead_ogs_each_obs(
                process,
                obs_point,
                which,
                time_steps,
                tecs=tecs,
                path_to_project=path_to_project,
                single_file=True,
                save_heads=True,
            )
            # recharge = getrecharge(path_to_project=path_to_project, name_of_project_ogs=name_of_project_ogs, time_steps=time_steps)
        elif which_data_to_plot == 1:
            fft_data = gethead_gw_model_each_obs(
                make_array_gw_model(
                    split_gw_model(
                        getlist_gw_model(
                            str(path_to_project)
                            + str(name_of_project_gw_model)
                            + "/H.OUT"
                        ),
                        index=2,
                    )
                ),
                convert_obs_list_to_index("obs_0990"),
                save_heads=False,
            )
            # recharge = getrecharge(path_to_project=path_to_project, name_of_project_ogs=name_of_project_ogs, time_steps=time_steps)
        elif which_data_to_plot == 3:
            # recharge = getrecharge(path_to_project=path_to_project, name_of_project_ogs=name_of_project_ogs, time_steps=time_steps)
            fft_data = recharge

    # convert recharge from list to array
    recharge = np.asarray([float(i) for i in recharge])
    return fft_data, recharge


# define the function to fit (linear aquifer model):
# a_d = np.mean(power_spectrum_result[:5])
def dupuit_fit(f_d, a_d, t_d):
    w_d = f_d * np.pi * 2
    return (
        (1.0 / a_d) ** 2
        * (
            (1.0 / (t_d * w_d))
            * np.tanh((1 + 1j) * np.sqrt(1.0 / 2 * t_d * w_d))
            * np.tanh((1 - 1j) * np.sqrt(1.0 / 2 * t_d * w_d))
        )
    ).real


# define the function to fit (linear aquifer model):
def linear_fit(f_l, a_l, t_l):
    w_l = f_l * np.pi * 2
    return 1.0 / (a_l ** 2 * (1 + t_l ** 2 * w_l ** 2))


# =============================================================================
# Calculate the discrete fourier transformation
# =============================================================================
def fft_psd(
    fft_data,
    recharge,
    threshold=1,
    aquifer_thickness=None,
    aquifer_length=None,
    distance_to_river=None,
    path_to_project="no_path_given",
    single_file="no_path_given",
    method="scipyffthalf",
    fit=False,
    savefig=False,
    saveoutput=True,
    dupuit=False,
    a_l=None,
    t_l=None,
    a_d=None,
    t_d=None,
    weights_l=[1, 1, 1, 1, 1, 1],
    weights_d=[1, 1, 1, 1, 1, 1],
    o_i="oi",
    time_step_size=None,
    windows=10,
    wiener_window=100,
    obs_point="no_obs_given",
    comment="",
    Ss_list=[],
    kf_list=[],
    obs_number=0,
    model_number=0,
    distance_to_river_list=0,
    target=False,
    ymin=None,
    ymax=None,
    xmin=None,
    xmax=None,
    a_of_x=False,
    a_alterna=False,
    detrend=False,
    shh_anal=False,
    Sy=None,
    T=None,
    anal_fit=None,
    anal_fit_norm=True,
    model_fit=True
):

    print("Function arguments:"
    +"\nthreshold: "    +str(threshold)
    +"\naquifer_thickness: " +str(aquifer_thickness)
    +"\naquifer_length: "   + str(aquifer_length)
    +"\ndistance_to_river: " +str(distance_to_river)
    +"\npath_to_project: " +  str(path_to_project)
    +"\nsingle_file: " +   str(single_file)
    +"\nmethod: "  + str( method)
    +"\nfit: "   + str(fit)
    +"\nsavefig: "  + str( savefig)
    +"\nsaveoutput: " + str(saveoutput)
    +"\ndupuit: "  + str( dupuit)
    +"\na_l: "   + str(a_l)
    +"\nt_l: "   + str(t_l)
    +"\na_d: "   + str(a_d)
    +"\nt_d: "   + str(t_d)
    +"\nweights_l: " +   str(weights_l)
    +"\nweights_d: "+    str(weights_d)
    +"\no_i: "   + str(o_i)
    +"\ntime_step_size: " +   str(time_step_size)
    +"\nwindows: "   + str(windows)
    +"\nwiener_window: "+    str(wiener_window)
    +"\nobs_point: "   + str(obs_point)
    +"\ncomment: "    +str(comment)
    +"\nSs_list: "   + str(Ss_list)
    +"\nkf_list: "   + str(kf_list)
    +"\nobs_number: "   + str(obs_number)
    +"\nmodel_number: "   + str(model_number)
    +"\ndistance_to_river_list: " +   str(distance_to_river_list)
    +"\ntarget: "  +  str(target)
    +"\nymin: "   + str(ymin)
    +"\nymax: "   + str(ymax)
    +"\nxmin: "   + str(xmin)
    +"\nxmax: "   + str(xmax)
    +"\na_of_x: "   + str(a_of_x)
    +"\na_alterna: "  +  str(a_alterna)
    +"\ndetrend: "   + str(detrend)
    +"\nshh_anal: " +   str(shh_anal)
    +"\nSy: " +   str(Sy)
    +"\nT: " +   str(T)
    )




    o_i_txt = ""
    threshold_txt = ""
    fit_txt = ""

    len_input = len(recharge)
    len_output = len(fft_data)

    # check if recharge and fft_data have the same length and erase values in the end
    if len(recharge) > len(fft_data):
        print(
            "The length of your input data is bigger than the length of you output data. Equalizing by deleting last entries from output data."
        )
        recharge = recharge[: len(fft_data)]
    elif len(recharge) < len(fft_data):
        print(
            "The length of your output data is bigger than the length of you input data. Equalizing by deleting last entries from input data."
        )
        fft_data = fft_data[: len(recharge)]

    # define the sampling frequency/time step
    # -------------------------------------------------------------------------
    sampling_frequency = 1.0 / time_step_size  # [Hz] second: 1, day: 1.1574074074074E-5

    # detrend input and output signal
    # -------------------------------------------------------------------------
    if detrend == True:
        print("Time series have been detrended.")
        recharge_detrend = signal.detrend(recharge, type="linear")
        fft_data_detrend = signal.detrend(fft_data, type="linear")
    else:
        print("Time series haven't been detrended.")
        recharge_detrend = recharge
        fft_data_detrend = fft_data




    # different methodologies for power spectral density
    # -------------------------------------------------------------------------

    if method == "scipyfftnormt":
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2 / T
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = (
            abs(fftpack.fft(recharge_detrend)[: len(fft_data_detrend) / 2]) ** 2
        ) / (len(fft_data) * time_step_size)
        power_spectrum_output = (
            abs(fftpack.fft(fft_data_detrend)[: len(fft_data_detrend) / 2]) ** 2
        ) / (len(fft_data) * time_step_size)
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = abs(fftpack.fftfreq(len(fft_data_detrend), time_step_size))[
            : len(fft_data_detrend) / 2
        ]
        if o_i == "i":
            power_spectrum_result = power_spectrum_input
        elif o_i == "o":
            power_spectrum_result = power_spectrum_output

    if method == "scipyfftnormn":
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2 / N
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = (
            abs(fftpack.fft(recharge_detrend)[: len(fft_data_detrend) / 2]) ** 2
        ) / len(fft_data)
        power_spectrum_output = (
            abs(fftpack.fft(fft_data_detrend)[: len(fft_data_detrend) / 2]) ** 2
        ) / len(fft_data)
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = abs(fftpack.fftfreq(len(fft_data_detrend), time_step_size))[
            : len(fft_data_detrend) / 2
        ]
        if o_i == "i":
            power_spectrum_result = power_spectrum_input
        elif o_i == "o":
            power_spectrum_result = power_spectrum_output

    if method == "scipyfftdouble":
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = (
            abs(fftpack.fft(recharge_detrend)[: len(fft_data_detrend) / 2]) ** 2
        ) * 2
        power_spectrum_output = (
            abs(fftpack.fft(fft_data_detrend)[: len(fft_data_detrend) / 2]) ** 2
        ) * 2
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = abs(fftpack.fftfreq(len(fft_data_detrend), time_step_size))[
            : len(fft_data_detrend) / 2
        ]
        if o_i == "i":
            power_spectrum_result = power_spectrum_input
        elif o_i == "o":
            power_spectrum_result = power_spectrum_output

    if method == "scipyrfft":
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = (abs(fftpack.rfft(recharge_detrend, len_input)) ** 2)[1:]
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (abs(fftpack.rfftfreq(len_output, time_step_size)))[1:]

    if method == "scipyrffthalf":
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = (
            (abs(fftpack.rfft(recharge_detrend, len_input))[: len_output / 2]) ** 2
        )[1:]
        power_spectrum_output = (
            (abs(fftpack.rfft(fft_data_detrend, len_output))[: len_output / 2]) ** 2
        )[1:]
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (
            abs(fftpack.rfftfreq(len_output, time_step_size))[: len_output / 2]
        )[1:]

    if method == "scipyfft":
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = (abs(fftpack.fft(recharge_detrend, len_input)) ** 2)[1:]
        power_spectrum_output = (abs(fftpack.fft(fft_data_detrend, len_output)) ** 2)[
            1:
        ]
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (abs(fftpack.fftfreq(len_output, time_step_size)))[1:]

    if method == "scipyffthalf":
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = (
            abs(
                (fftpack.fft(recharge_detrend, len_input)[: len_output / 2])
            )
            ** 2
        )[1:]
        power_spectrum_output = (
            abs(
                (fftpack.fft(fft_data_detrend, len_output)[: len_output / 2])
            )
            ** 2
        )[1:]
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (
            abs(fftpack.fftfreq(len_output, time_step_size))[: len_output / 2]
        )[1:]

        if o_i == "i":
            power_spectrum_result = power_spectrum_input
            o_i_txt = "in_"
        elif o_i == "o":
            power_spectrum_result = power_spectrum_output
            o_i_txt = "out_"

    if method == "scipyffthalfpi":
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = (
            abs(
                (fftpack.fft(recharge_detrend, len_input)[: len_output / 2])
                / 2.0
                / np.pi
            )
            ** 2
        )[1:]
        power_spectrum_output = (
            abs(
                (fftpack.fft(fft_data_detrend, len_output)[: len_output / 2])
                / 2.0
                / np.pi
            )
            ** 2
        )[1:]
        # power_spectrum_input = np.asarray([i/(2*np.pi) for i in power_spectrum_input])
        # power_spectrum_output = np.asarray([i/(2*np.pi) for i in power_spectrum_output])
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (
            abs(fftpack.fftfreq(len_output, time_step_size))[: len_output / 2]
        )[1:]

        if o_i == "i":
            power_spectrum_result = power_spectrum_input
            o_i_txt = "in_"
        elif o_i == "o":
            power_spectrum_result = power_spectrum_output
            o_i_txt = "out_"


    if method == "autocorrelation":
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        autocorr_in = np.correlate(recharge, recharge, mode="full")
        autocorr_out = np.correlate(fft_data, fft_data, mode="full")
        power_spectrum_input = autocorr_in[len(autocorr_in) / 2 :]
        power_spectrum_output = autocorr_out[len(autocorr_out) / 2 :]
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (
            abs(fftpack.fftfreq(autocorr_in, time_step_size))[: len_output / 2]
        )[1:]

        if o_i == "i":
            power_spectrum_result = power_spectrum_input
            o_i_txt = "in_"
        elif o_i == "o":
            power_spectrum_result = power_spectrum_output
            o_i_txt = "out_"

    if method == "scipywelch":
        # =========================================================================
        # method x: scipy.signal.welch
        #           https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html#r145
        # =========================================================================
        frequency_input, power_spectrum_input = signal.welch(
            recharge_detrend, sampling_frequency, nperseg=1000, window="hamming"
        )
        frequency_output, power_spectrum_output = signal.welch(
            fft_data_detrend, sampling_frequency, nperseg=1000, window="hamming"
        )
        frequency_output = frequency_output[1:]
        frequency_input = frequency_input[1:]
        power_spectrum_result = (
                power_spectrum_output / power_spectrum_input
        )[1:]
        if o_i == "i":
            power_spectrum_result = power_spectrum_input[1:]
        elif o_i == "o":
            power_spectrum_result = power_spectrum_output[1:]

#    if method == "pyplotwelch":
#        # =========================================================================
#        # method x: Pyplot PSD by Welch
#        #           https://matplotlib.org/api/_as_gen/matplotlib.pyplot.psd.html
#        # =========================================================================
#        power_spectrum_input, frequency_input = plt.psd(
#            recharge_detrend, Fs=sampling_frequency
#        )
#        power_spectrum_output, frequency_output = plt.psd(
#            fft_data_detrend, Fs=sampling_frequency
#        )
#        # delete first value (which is 0) because it makes trouble with fitting
#        frequency_output = frequency_output  # [1:]
#        frequency_input = frequency_input  # [1:]
#        power_spectrum_result = power_spectrum_output / power_spectrum_input  # [1:]
#        if o_i == "i":
#            power_spectrum_result = power_spectrum_input[1:]
#        elif o_i == "o":
#            power_spectrum_result = power_spectrum_output[1:]

    if method == "scipyperio":
        # =========================================================================
        # method x: Scipy.signal.periodogram
        #           https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.signal.periodogram.html
        # =========================================================================
        frequency_input, power_spectrum_input = signal.periodogram(
            recharge_detrend, fs=sampling_frequency
        )
        frequency_output, power_spectrum_output = signal.periodogram(
            fft_data_detrend, fs=sampling_frequency
        )
        frequency_output = frequency_output[1:]
        frequency_input = frequency_input[1:]
        power_spectrum_result = (power_spectrum_output / power_spectrum_input)[1:]
        if o_i == "i":
            power_spectrum_result = power_spectrum_input[1:]
        elif o_i == "o":
            power_spectrum_result = power_spectrum_output[1:]

#    if method == "spectrumperio":
#        # =========================================================================
#        # method x: Spectrum.periodogram
#        #           http://thomas-cokelaer.info/software/spectrum/html/user/ref_fourier.html#spectrum.periodogram.Periodogram
#        # =========================================================================
#        from spectrum import WelchPeriodogram
#
#        power_spectrum_input, empty = WelchPeriodogram(recharge_detrend, 256)
#        plt.close()
#        frequency_input = power_spectrum_input[1]
#        frequency_input = frequency_input[1:]
#        power_spectrum_input = power_spectrum_input[0]
#        power_spectrum_output, empty = WelchPeriodogram(fft_data_detrend, 256)
#        plt.close()
#        frequency_output = power_spectrum_output[1]
#        frequency_output = frequency_output[1:]
#        power_spectrum_output = power_spectrum_output[0]
#        power_spectrum_result = (power_spectrum_output / power_spectrum_input)[1:]
#        if o_i == "i":
#            power_spectrum_result = power_spectrum_input[1:]
#        elif o_i == "o":
#            power_spectrum_result = power_spectrum_output[1:]

    """
    Further methods, not working or still under construction
    elif method == 'spectrum_sperio':
        # =========================================================================
        # method x: Spectrum.speriodogram
        #           http://thomas-cokelaer.info/software/spectrum/html/user/ref_fourier.html#spectrum.periodogram.Periodogram
        # =========================================================================
        from spectrum import speriodogram
        power_spectrum_input = speriodogram(recharge_detrend,
                                            detrend = False,
                                            sampling = sampling_frequency)
        power_spectrum_output = speriodogram(fft_data_recharge,
                                            detrend = False,
                                            sampling = sampling_frequency)
        power_spectrum_result = power_spectrum_output / power_spectrum_input

    elif method == 'correlation':
        # =========================================================================
        # method x: CORRELOGRAMPSD.periodogram
        #           http://thomas-cokelaer.info/software/spectrum/html/user/ref_fourier.html#spectrum.periodogram.Periodogram
        # =========================================================================
        from spectrum import CORRELOGRAMPSD
        tes = CORRELOGRAMPSD(recharge_detrend, recharge_detrend, lag=15)
        psd = tes[len(tes)/2:]
    """

    # delete values with small frequencies at given threshold
    i = 0
    for i, value in enumerate(frequency_input):
        if value > threshold:
            cutoff_index = i
            print(
                "PSD was cut by threshold. Remaining data points: " + str(cutoff_index)
            )
            for j in range(i, len(frequency_input)):
                frequency_input = np.delete(frequency_input, i)
                power_spectrum_result = np.delete(power_spectrum_result, i)
            break

    # plot the resulting power spectrum
    # -------------------------------------------------------------------------
    font = {"family": "normal", "weight": "normal", "size": 15}
    plt.rc("font", **font)
    plt.rc("legend",fontsize=10)
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(
        left=None, bottom=0.25, right=None, top=None, wspace=None, hspace=None
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1/s")
    if ymin != None and ymax != None:
        ax.set_ylim(ymin, ymax)
    if xmin != None and xmax != None:
        ax.set_xlim(xmin, xmax)

    # ax.plot(freq_month[ind],psd)
    ax.plot(frequency_input, power_spectrum_result, label="PSD")

    # power spectral density analytical
    # -------------------------------------------------------------------------
    if shh_anal == True:
        print("!! Calculating analytical psd...")
        print("Input parameters are: "
              + "\nSy: "
              + str(Sy)
              + "\nT: "
              + str(T)
              + "\nLocation: "
              + str(aquifer_length - distance_to_river)
              + "\nA. Length: "
              + str(aquifer_length)
              )
        power_spectrum_output_anal = shh_analytical((frequency_input, power_spectrum_input), Sy, T, aquifer_length - distance_to_river, aquifer_length)



    ax.set_title(
        "Power Spectral Density for observation point "
        + str(obs_point)
        + "\n"
        + "method: "
        + str(method)
    )
    ax.grid(color="grey", linestyle="--", linewidth=0.5, which="both")

    # =========================================================================
    # Fit the power spectrum
    # =========================================================================

    if fit == True:

        # employ a filter on the spectrum to optimize the fit
        # ---------------------------------------------------------------------
        # method a: savgol
        window_size = 11
        # window_size = np.around((len(power_spectrum_result)/windows),0)
        # if window_size % 2 == 0:
        #    window_size = window_size + 1
        # elif window_size < 2:
        #    window_size = 2
        power_spectrum_result_filtered = signal.savgol_filter(
            power_spectrum_result, window_size, 5
        )
        # method b: wiener
        # power_spectrum_result_filtered = signal.wiener(power_spectrum_result, wiener_window)
        # power_spectrum_result_filtered = moving_average(power_spectrum_result, 3)
        # ax.plot(frequency_input[:len(power_spectrum_result_filtered)], power_spectrum_result_filtered, label='filtered PSD')

        # =====================================================================
        # analytical solution by Liang and Zhang 2013
        # =====================================================================

        if anal_fit == True:
            from fit_analytical_psd import shh_analytical_fit
            popt, pcov = shh_analytical_fit(power_spectrum_input, power_spectrum_result, frequency_input, aquifer_length-distance_to_river, aquifer_length, m=5, n=5, norm=False)
            popt[0] = abs(popt[0])
            popt[1] = abs(popt[1])

            print("Inferred aquifer parameters: Storativity, Transmissivity: " + str(popt[0]) + ", " + str(popt[1]))
            S_anal = popt[0]
            T_anal = popt[1]
            kf_anal = popt[1] / aquifer_thickness
            D_anal = popt[1] /popt[0]

            if popt[0] == 1.0 and popt[1] == 1.0:
                print("T and S have been set to nan.")
                T_anal, S_anal = np.nan, np.nan


            output_anal = (
                    "Analytical fit:\n "
                    + "T [m2/s]: "
                    + "%0.4e" % T_anal
                    + "\n  "
                    + "S [-]: "
                    + "%0.4e" % S_anal
                    + "\n  "
                    + "kf [m/s]: "
                    + "%0.4e" % kf_anal
                    + "\n  "
                    + "D [m2/s]: "
                    + "%0.4e" % D_anal
                )

            input_param = (
                    "OGS input parameter:\n "
                    + "T [m2/s]: "
                    + "%0.4e" % T
                    + "\n  "
                    + "S [-]: "
                    + "%0.4e" % Sy
                )
            fig_txt_1 = tw.fill(str(output_anal), width=145)
            fig_txt_2 = tw.fill(str(input_param), width=145)
            text = []
            text.append(fig_txt_1)
            text.append(fig_txt_2)
            fig_txt = '\n\n'.join(text for text in text)

            plt.figtext(
                0.5,
                0.05,
                fig_txt,
                horizontalalignment="center",
                bbox=dict(boxstyle="square", facecolor="#F2F3F4", ec="1", pad=0.8, alpha=1),
            )
<<<<<<< HEAD


            ax.plot(frequency_input, shh_analytical((frequency_input,power_spectrum_input), popt[0], popt[1], aquifer_length-distance_to_river, aquifer_length, m=5, n=5, norm=anal_fit_norm), label="analytical fit", color="red")

=======
                        
            
            ax.plot(frequency_input, shh_analytical((frequency_input,power_spectrum_input), popt[0], popt[1], aquifer_length-distance_to_river, aquifer_length, m=5, n=5, norm=anal_fit_norm), label="analytical fit", color="red") #, ls="", marker="*")
            
>>>>>>> 065ad4a9bd7535d37e91a626fd20c12459f267a8
            T_l = np.nan
            kf_l = np.nan
            S_l = np.nan
            Ss_l = np.nan
            D_l = np.nan
            t_l = np.nan
            a_l = np.nan
            T_d = np.nan
            kf_d = np.nan
            S_d = np.nan
            Ss_d = np.nan
            D_d = np.nan
            t_d = np.nan
            a_d = np.nan

<<<<<<< HEAD

=======
            
            
>>>>>>> 065ad4a9bd7535d37e91a626fd20c12459f267a8
        if model_fit == True:
            # =====================================================================
            # linear model
            # =====================================================================
            # least squares automatic fit for linear aquifer model (Gelhar, 1993):
            # abs(H_h(w))**2 = 1 / (a**2 * ( 1 + ((t_l**2) * (w**2))))
            # ---------------------------------------------------------------------

            if a_l == None and t_l == None:
                # make an initial guess for a_l, and t_l
                initial_guess = np.array([1e-15, 40000])

                # generate a weighing array
                # ---------------------------------------------------------------------
                # based on dividing the data into segments
                sigma_l = []
                data_per_segment = len(power_spectrum_result) / len(weights_l)
                for weight_l in weights_l:
                    sigma_l = np.append(sigma_l, np.full(data_per_segment, weight_l))
                if len(power_spectrum_result) % len(weights_l) != 0:
                    for residual in range(len(power_spectrum_result) % len(weights_l)):
                        sigma_l = np.append(sigma_l, weights_l[-1])

                try:
                    # perform the fit
                    popt_l, pcov_l = optimization.curve_fit(
                        linear_fit,
                        frequency_input,
                        power_spectrum_result,
                        p0=initial_guess,
                        sigma=sigma_l,
                    )
                    # abs to avoid negative values from optimization
                    t_l = abs(popt_l[1])
                    a_l = abs(popt_l[0])

                    # Plot the linear fit model
                    # ---------------------------------------------------------------------
                    linear_model = []
                    # fitting model for the linear reservoir (Gelhar, 1993)
                    for i in range(0, len(frequency_input)):
                        line = linear_fit(frequency_input[i], a_l, t_l)
                        linear_model.append(line)
                    ax.plot(frequency_input, linear_model, label="linear model")

                    # plot the linear model with input parameters of ogs
                    if target == True:
                        if a_of_x == True:
                            print("Calculating parameters for target model for parameter 'a' in dependence on x"
                                  + "\nSs: " + str(Ss_list[model_number])
                                  + "\nkf: " + str(kf_list[model_number])
                                  + "\naquifer length: " + str(aquifer_length)
                                  + "\naquifer thickness: " + str(aquifer_thickness)
                                  + "\nmodel: " + "linear"
                                  + "\ndistance to river: " + str(distance_to_river_list)
                                  )
                            params_real = calc_aq_param(
                                Ss_list[model_number],
                                kf_list[model_number],
                                aquifer_length,
                                aquifer_thickness,
                                model="linear",
                                distance=distance_to_river_list[obs_number])
                        else:
                            print("Calculating parameters for target model for parameter 'a' INdependent on x"
                                  + "\nSs: " + str(Ss_list[model_number])
                                  + "\nkf: " + str(kf_list[model_number])
                                  + "\naquifer length: " + str(aquifer_length)
                                  + "\naquifer thickness: " + str(aquifer_thickness)
                                  + "\nmodel: " + "linear"
                                  )
                            params_real = calc_aq_param(
                                Ss_list[model_number],
                                kf_list[model_number],
                                aquifer_length,
                                aquifer_thickness,
                                model="linear")
                        ax.plot(
                            frequency_input,
                            [
                                linear_fit(
                                    a_l=params_real[4],
                                    t_l=params_real[5],
                                    f_l=frequency_input[i],
                                )
                                for i in range(0, len(frequency_input))
                            ],
                            label="linear model, target",
                        )

                    # calculate aquifer parameters
                    # ---------------------------------------------------------------------
                    if a_of_x == True:
                        print("Calculation of T in dependence on location in aquifer.")
                        T_l = (
                            a_l
                            * aquifer_length ** 2
                            * (1 - ((float(distance_to_river) / aquifer_length) - 1)) ** 4
                            )
                        print("T_l = (a_l* aquifer_length ** 2 * (1 - ((float(distance_to_river) / aquifer_length) - 1)) ** 4)")
                        print("T_l = " + str(a_l) + " * " + str(aquifer_length) + " ** 2 * (1 - ((" + str(distance_to_river) + ") / " + str(aquifer_length) + ") - 1)) ** 4)")
                    if a_of_x == False:
                        T_l = a_l * aquifer_length**2 / 3.
                        print("T_l = ", a_l, "*", aquifer_length, "**2 / 3.")
                        print("'T_l = ', a_l, '*', aquifer_length, '**2 / 3.'")
                    kf_l = T_l / aquifer_thickness
                    S_l = a_l * t_l
                    Ss_l = S_l / aquifer_thickness
                    D_l = T_l / S_l
                    #D_l = aquifer_length ** 2 / (3.0 * t_l)
                    # D_l = aquifer_length**2 * 4 / (np.pi**2 * t_l)
                    print("kf_l = ", T_l, "/", aquifer_thickness)
                    print("'kf_l = ', T_l, '/', aquifer_thickness")
                    print("S_l = ", a_l, "*", t_l)
                    print("'S_l = ', a_l, '*', t_l")
                    print("Ss_l = ", S_l, "/", aquifer_thickness)
                    print("'Ss_l = ', S_l, '/', aquifer_thickness")
                    print("D_l = ", aquifer_length, "**2 / (3 * ", t_l, ")")
                    print("'D_l = ', aquifer_length, '**2 / (3 * ', t_l,')'")
                    output_l = (
                        "Linear model:\n "
                        + "T [m2/s]: "
                        + "%0.4e" % T_l
                        + "\n  "
                        + "Ss [1/m]: "
                        + "%0.4e" % Ss_l
                        + "\n  "
                        + "kf [m/s]: "
                        + "%0.4e" % kf_l
                        + "\n  "
                        + "D [m2/s]: "
                        + "%0.4e" % D_l
                        + "\n  "
                        + "a: "
                        + "%0.4e" % a_l
                        + "\n  "
                        + "t_c [s]: "
                        + "%0.4e" % t_l
                    )
                    print(output_l)
                    fig_txt = tw.fill(output_l, width=250)
                except RuntimeError:

                    print(
                        "Automatic linear model fit failed... Provide a_l and t_l manually!"
                    )
                    # calculate aquifer parameters
                    # ---------------------------------------------------------------------
                    T_l = np.nan
                    kf_l = np.nan
                    S_l = np.nan
                    Ss_l = np.nan
                    D_l = np.nan
                    t_l = np.nan
                    a_l = np.nan
                    # D_l = aquifer_length**2 * 4 / (np.pi**2 * t_l)
                    output_l = ""
                    print(output_l)
            else:
                # Plot the linear fit model
                # ---------------------------------------------------------------------
                linear_model = []
                # fitting model for the linear reservoir (Gelhar, 1993)
                for i in range(0, len(frequency_input)):
                    line = linear_fit(frequency_input[i], a_l, t_l)
                    linear_model.append(line)
                ax.plot(frequency_input, linear_model, label="linear model")

                # calculate aquifer parameters
                # ---------------------------------------------------------------------
                if a_of_x == True:
                    print("Calculation of T in dependence of location in aquifer.")
                    T_l = (
                        a_l
                        * aquifer_length ** 2
                        * (1 - ((float(distance_to_river) / aquifer_length) - 1)) ** 4
                        )
                print("T_l = (a_l* aquifer_length ** 2 * (1 - ((float(distance_to_river) / aquifer_length) - 1)) ** 4)")
                print("T_l = " + str(a_l) + " * " + str(aquifer_length) + " ** 2 * (1 - ((" + str(distance_to_river) + ") / " + str(aquifer_length) + ") - 1)) ** 4)")
                if a_of_x == False:
                    T_l = a_l * aquifer_length**2 / 3.
                kf_l = T_l / aquifer_thickness
                S_l = a_l * t_l
                Ss_l = S_l / aquifer_thickness
                D_l = aquifer_length ** 2 / (3 * t_l)
                # D_l = aquifer_length**2 * 4 / (np.pi**2 * t_l)
                print("T_l = ", a_l, "*", aquifer_length, "**2 / 3.")
                print("'T_l = ', a_l, '*', aquifer_length, '**2 / 3.'")
                print("kf_l = ", T_l, "/", aquifer_thickness)
                print("'kf_l = ', T_l, '/', aquifer_thickness")
                print("S_l = ", a_l, "*", t_l)
                print("'S_l = ', a_l, '*', t_l")
                print("Ss_l = ", S_l, "/", aquifer_thickness)
                print("'Ss_l = ', S_l, '/', aquifer_thickness")
                print("D_l = ", aquifer_length, "**2 / (3 * ", t_l, ")")
                print("'D_l = ', aquifer_length, '**2 / (3 * ', t_l,')'")
                output_l = (
                    "Linear model:\n "
                    + "T [m2/s]: "
                    + "%0.4e" % T_l
                    + "\n  "
                    + "Ss [1/m]: "
                    + "%0.4e" % Ss_l
                    + "\n  "
                    + "kf [m/s]: "
                    + "%0.4e" % kf_l
                    + "\n  "
                    + "D [m2/s]: "
                    + "%0.4e" % D_l
                    + "\n  "
                    + "a: "
                    + "%0.4e" % a_l
                    + "\n  "
                    + "t_c [s]: "
                    + "%0.4e" % t_l
                )
                print(output_l)
                fig_txt = tw.fill(output_l, width=250)

            # =====================================================================
            # Dupuit Model
            # =====================================================================
            # Step 5: least squares automatic fit for Dupuit-Aquifer model
            # (e.g. Gelhar and Wilson, 1974):
            # abs(H_h(w))**2 = (b/E)**2 * ( (1/O)*tanh)((1+j)*sqrt(1/2*O))*tanh((1-j)*sqrt(1/2*O))
            # O = td * w
            # E = x - x_o    distance from river
            # ---------------------------------------------------------------------

            if a_d == None and t_d == None and dupuit == True:
                # make an initial guess for a_l, and t_l
                initial_guess = np.array([0.98e-15, 2000000])

                # generate a weighing array
                # ---------------------------------------------------------------------
                # based on dividing the data into segments
                sigma_d = []
                # weights = [1,1,1] # give the weights for each segment, amount of values specifies the amount of segments
                data_per_segment = len(power_spectrum_result) / len(weights_d)
                for weight_d in weights_d:
                    sigma_d = np.append(sigma_d, np.full(data_per_segment, weight_d))
                if len(power_spectrum_result) % len(weights_d) != 0:
                    for residual in range(len(power_spectrum_result) % len(weights_d)):
                        sigma_d = np.append(sigma_d, weights_d[-1])

                try:
                    # perform the fit
                    popt_d, pcov_d = optimization.curve_fit(
                        dupuit_fit,
                        frequency_input,
                        power_spectrum_result,
                        p0=initial_guess,
                        sigma=sigma_d,
                    )
                    # abs to avoid negative values
                    # a_d = popt_d[0]
                    # a_d = popt_d[0]
                    a_d = popt_d[0]
                    t_d = popt_d[1]

                except RuntimeError:
                    T_d, kf_d, Ss_d, D_d = np.nan, np.nan, np.nan, np.nan
                    print("Dupuit fit failed... Provide a_d and a_t manually!")

                # assign nan to alls parameters if duptui model is not used
            else:
                T_d, kf_d, Ss_d, D_d = np.nan, np.nan, np.nan, np.nan

            # Plot the Dupuit model
            # ---------------------------------------------------------------------
            try:
                dupuit_model = []
                # fitting model for the linear reservoir (Gelhar, 1993)
                for i in range(0, len(frequency_input)):
                    line = dupuit_fit(frequency_input[i], a_d, t_d)
                    dupuit_model.append(line)
                ax.plot(frequency_input, dupuit_model, label="Dupuit model")

                # plot the dupuit model with input parameters of ogs
                if target == True:
                    print("Calculating parameters for target model for parameter 'a' in dependence on x"
                                  + "\nSs: " + str(Ss_list[model_number])
                                  + "\nkf: " + str(kf_list[model_number])
                                  + "\naquifer length: " + str(aquifer_length)
                                  + "\naquifer thickness: " + str(aquifer_thickness)
                                  + "\nmodel: " + "dupuit"
                                  + "\ndistance to river: " + str(distance_to_river_list)
                                  + "a_alterna: " + str(a_alterna)
                                  )
                    params_real = calc_aq_param(
                        Ss_list[model_number],
                        kf_list[model_number],
                        aquifer_length,
                        aquifer_thickness,
                        model="dupuit",
                        distance=distance_to_river_list[obs_number],
                        a_alterna=a_alterna
                    )
                    ax.plot(
                        frequency_input,
                        [
                            dupuit_fit(
                                a_d=params_real[4], t_d=params_real[5], f_d=frequency_input[i]
                            )
                            for i in range(0, len(frequency_input))
                        ],
                        label="dupuit model, target",
                    )

                # calculate aquifer parameters
                # ---------------------------------------------------------------------
                #print("calculation of T with new formula")
                #T_d = (
                #    a_d
                #    * aquifer_length ** 2
                #    * (1 - ((float(distance_to_river) / aquifer_length) - 1)) ** 4
                #)


                # method from Gelhar 1974, beta = pi^2/4
                if a_alterna == True:
                    T_d = a_d * aquifer_length**2 * 4 / np.pi**2
                else:
                    T_d = a_d * aquifer_thickness * distance_to_river
                kf_d = T_d / aquifer_thickness
                S_d = t_d * T_d / aquifer_length ** 2
                Ss_d = S_d / aquifer_thickness
                D_d = T_d / S_d
                print("T_d = ", a_d, "*", aquifer_thickness, "*", distance_to_river)
                print("'T_d = ', a_d, '*', aquifer_thickness, '*', distance_to_river")
                print("kf_d = ", T_d, "/", aquifer_thickness)
                print("'kf_d = ', T_d, '/', aquifer_thickness")
                print("S_d = ", t_d, "*", T_d, "/", aquifer_length, "**2")
                print("'S_d = ', t_d, '*', T_d, '/', aquifer_length, '**2'")
                print("Ss_d = ", S_d, "/", aquifer_thickness)
                print("'Ss_d = ', S_d, '/', aquifer_thickness")
                print("D_d = ", T_d, "/", S_d)
                print("'D_d = ', T_d, '/', S_d")
                output_d = (
                    "Dupuit model: \n"
                    + "T [m2/s]: "
                    + "%0.4e" % T_d
                    + "\n  "
                    + "Ss [1/m]: "
                    + "%0.4e" % Ss_d
                    + "\n  "
                    + "kf [m/s]: "
                    + "%0.4e" % kf_d
                    + "\n  "
                    + "D [m2/s]: "
                    + "%0.4e" % D_d
                    + "\n  "
                    + "a: "
                    + "%0.4e" % a_d
                    + "\n  "
                    + "t_c [s]: "
                    + "%0.4e" % t_d
                )
                print(output_d)
                fig_txt = tw.fill(str(output_l) + "\n" + str(output_d), width=145)

            except TypeError:
                print("Automatic Dupuit-model fit failed... Provide a_d and t_d manually.")
                T_d, kf_d, Ss_d, D_d = np.nan, np.nan, np.nan, np.nan
                fig_txt = tw.fill(str(output_l), width=200)

            # annotate the figure
            # fig_txt = tw.fill(tw.dedent(output), width=120)
            plt.figtext(
                0.5,
                0.05,
                fig_txt,
                horizontalalignment="center",
                bbox=dict(boxstyle="square", facecolor="#F2F3F4", ec="1", pad=0.8, alpha=1),
            )

        #    if a_d == None and t_d == None and dupuit == True:



    # plot the target analytical power spectrum based on input parameters from ogs model runs
    # ---------------------------------------------------------------------
<<<<<<< HEAD
    if o_i == "o":
        ax.plot(frequency_input, power_spectrum_output_anal, label="analytical target", color="green", marker="o", ls="", markersize=0.5)
    if o_i == "oi":
        power_spectrum_result_anal = power_spectrum_output_anal / power_spectrum_input
        ax.plot(frequency_input, power_spectrum_result_anal, label="analytical, target")
=======
    if anal_fit == True:
        if o_i == "o":
            ax.plot(frequency_input, power_spectrum_output_anal, label="analytical target", color="green", marker="o", ls="", markersize=0.5)
        if o_i == "oi":    
            power_spectrum_result_anal = power_spectrum_output_anal / power_spectrum_input
            ax.plot(frequency_input, power_spectrum_result_anal, label="analytical, target")
>>>>>>> 065ad4a9bd7535d37e91a626fd20c12459f267a8



    plt.legend(loc="best")
    #plt.show()
    if savefig == True:
        if fit == True:
            fit_txt = "fit_"
        if threshold != 1:
            threshold_txt = str(threshold) + "_"
        path_name_of_file_plot = (
            str(path_to_project)
            + "/"
            + str(comment)
            + "PSD_"
            + fit_txt
            + o_i_txt
            + threshold_txt
            + str(method)
            + "_"
            + str(os.path.basename(str(path_to_project)))
            + "_"
            + str(obs_point)
            + ".png"
        )
        print("Saving figure " + str(path_name_of_file_plot[-30:]))
        fig.savefig(path_name_of_file_plot)
    fig.clf()
    plt.close(fig)

    path_name_of_file_plot = (
        str(path_to_project)
        + "/"
        + str(comment)
        + "PSD_"
        + fit_txt
        + o_i_txt
        + threshold_txt
        + str(method)
        + "_"
        + str(os.path.basename(str(path_to_project)[:-1]))
        + "_"
        + str(obs_point)
        + ".png"
    )
    if fit == False:
        T_l = np.nan
        kf_l = np.nan
        S_l = np.nan
        Ss_l = np.nan
        D_l = np.nan
        t_l = np.nan
        a_l = np.nan
        T_d = np.nan
        kf_d = np.nan
        S_d = np.nan
        Ss_d = np.nan
        D_d = np.nan
        t_d = np.nan
        a_d = np.nan
        T_anal = np.nan
        S_anal = np.nan
        
    if fit == True and saveoutput == True:
        with open(str(path_to_project) + "/PSD_output.log", "a") as file:
            file.write(
                str(datetime.datetime.now())
                + " "
                + method
                + " "
                + str(T_l)
                + " "
                + str(kf_l)
                + " "
                + str(Ss_l)
                + " "
                + str(D_l)
                + " "
                + str(a_l)
                + " "
                + str(t_l)
                + " "
                + str(T_d)
                + " "
                + str(kf_d)
                + " "
                + str(Ss_d)
                + " "
                + str(D_d)
                + " "
                + str(a_d)
                + " "
                + str(t_d)
                + " "
                + str(path_name_of_file_plot)
                + "\n"
            )
        file.close()
    print("###################################################################")
    return (
        T_anal,
        S_anal,
        T_l,
        kf_l,
        Ss_l,
        D_l,
        a_l,
        t_l,
        T_d,
        kf_d,
        Ss_d,
        D_d,
        a_d,
        t_d,
        power_spectrum_output,
        power_spectrum_input,
        power_spectrum_output,
        power_spectrum_result,
        frequency_input,
    )
