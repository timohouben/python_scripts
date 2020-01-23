import numpy as np

from plot_power_spectra import plot_spectrum
from power_spectrum import power_spectrum
import pandas as pd

birkach = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/birkach_quadratic.txt"
stegaurach = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/stegaurach_quadratic.txt"
strullendorf_west = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/strull_west_quadratic.txt"
strullendorf_nord = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/strull_nord_quadratic.txt"
xs = [4400,600,7000,5100]
Ls = [4500,3000,8000,7800]
names = [birkach, stegaurach, strullendorf_west, strullendorf_nord]
artificial_data = False
m = None
n = None
norm = False
convergence = 0.01
time_step_size = 86400


for name, L, x in zip(names, Ls, xs):
    gw_head_df = pd.read_csv(
        name,
        sep=" ",
        header=None,
        names=["date", "gw_head"],
    )

    # load recharge data
    recharge_df = pd.read_csv(
        "/Users/houben/phd/mHM/GW_recharge_main/4348000.0_5454000.0_recharge.txt",
        sep=" ",
        header=None,
        names=["date", "recharge"],
    )

    # clip dataframe for min max of gw head time series
    combined_df = pd.merge(recharge_df, gw_head_df, how="inner", on=["date"])

    recharge_time_series = combined_df["recharge"].tolist()
    # convert mm/d to recharge along the aquifer in m2/s
    recharge_time_series = [i / 86400 / 1000 * L for i in recharge_time_series]

    head_time_series = combined_df["gw_head"].tolist()


    #'''
    #################################################
    #### TAKE ARTIFICIAL DATA TO PROOF THE SCRIPT
    head_time_series = np.loadtxt(
        "/Users/houben/phd/modelling/20190318_spectral_analysis_homogeneous/models_test/1001_24.1127_5.00e-01_8.00e-02/head_ogs_obs_00200_mean.txt"
    )
    recharge_time_series = np.loadtxt(
        "/Users/houben/phd/modelling/20190318_spectral_analysis_homogeneous/models_test/1001_24.1127_5.00e-01_8.00e-02/rfd_curve#1.txt"
    )

    from transect_plot import extract_timeseries, plot_head_timeseries_vs_recharge, extract_rfd
    path_to_project = "/Users/houben/phd/modelling/20190318_spectral_analysis_homogeneous/models_test/1001_24.1127_5.00e-01_8.00e-02"
    #time_time_series, recharge_time_series = extract_rfd(path=path_to_project, rfd=1)

    recharge_time_series = recharge_time_series * L
    L = 1000
    x = 200
    artificial_data = True
    #################################################
    #'''

    frequency_Shh_Sww, Shh_Sww = power_spectrum(recharge, gw_head, time_step_size, o_i="oi")

    frequency_Sww, Sww = power_spectrum(recharge, gw_head, time_step_size, o_i="i")

    frequency_Shh, Shh = power_spectrum(recharge, gw_head, time_step_size, o_i="o")

    # cut the power spectrum
    # cut_value of frequency
    #cut_freq = 1e-4
    #cut_array = np.less(frequency, cut_freq)
    #power_spectrum_input = power_spectrum_input[cut_array]
    #power_spectrum_output = power_spectrum_output[cut_array]
    #power_spectrum_result = power_spectrum_result[cut_array]
    #frequency = frequency[cut_array]


    from shh_analytical import shh_analytical, shh_analytical_fit

    # calculate the power spectrum: Shh/Sww, output/input to PLOT only!
    frequency_oi, Shh_Sww = power_spectrum(
        input=recharge_time_series,
        output=head_time_series,
        time_step_size=time_step_size,
        method="scipyffthalf",
        o_i="oi",
        )

    # calculate the power spectrum: Shh, output to FIT with analy solution only!
    frequency, Shh = power_spectrum(
        input=recharge_time_series,
        output=head_time_series,
        time_step_size=time_step_size,
        method="scipyffthalf",
        o_i="o",
        )

    frequency, Sww = power_spectrum(
        input=recharge_time_series,
        output=head_time_series,
        time_step_size=time_step_size,
        method="scipyffthalf",
        o_i="i",
        )
    # fit the power spectrum with the analytical solution
    try:
        popt, pcov = shh_analytical_fit(
            Sww=Sww,
            Shh=Shh,
            f=frequency_Shh,
            x=x,
            m=m,
            n=n,
            L=L,
            norm=False,
            convergence=convergence,
        )
    except RuntimeError:
        print("Optimal parameters not found...")
        popt, pcov = [np.nan, np.nan], [[np.nan, np.nan], [np.nan, np.nan]]
        print("popt and pcov have been set to np.nan")
    except ValueError:
        print("either ydata or xdata contain NaNs, or if incompatible options are used")
        popt, pcov = [np.nan, np.nan], [[np.nan, np.nan], [np.nan, np.nan]]
    except OptimizeWarning:
        print("Covariance of the parameters could not be estimated.")
        #popt, pcov = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]
    if artificial_data == True:
        break

print(popt[0])
print(popt[1])





'''



###########################################
    popt, pcov = shh_analytical_fit(
        Sww=power_spectrum_input,
        Shh=power_spectrum_output,
        f=frequency,
        x=x,
        L=L,
        m=m,
        n=n,
        norm=norm,
        convergence=convergence,
    )

    Sy = abs(popt[0])
    T = abs(popt[1])
    X = (frequency, power_spectrum_input)

    spectrum_fit = shh_analytical(
        X, Sy, T, x, L, m=None, n=None, norm=norm, convergence=0.01
    )

    # plot only spectrum of Shh
    plot_spectrum(
        [power_spectrum_output],
        frequency,
        heading=name,
        labels=["power_spec_out"],
        path="/Users/houben/phd/application_spectral_analysis/main/gwms/data",
        linestyle=["-"],
        marker=[""],
        lims=[(2e-9,7e-6),(10e-6,10e6)],
    )

    T = T/1000
    Sy = Sy/1000

    figtxt = "T = " + str(T) + ", S = " + str(Sy) + "\n" + "L = " + str(L) + ", x = " + str(x)

    plot_spectrum(
        np.vstack((power_spectrum_output, spectrum_fit)),
        frequency,
        heading=name,
        labels=["power_spec_out", "analytical fit"],
        path="/Users/houben/phd/application_spectral_analysis/main/gwms/data",
        linestyle=["-", " "],
        marker=["", "*"],
        figtxt=figtxt,
        lims=[(2e-9,7e-6),(10e-6,10e6)],
    )

    if artificial_data == True:
        break
'''
