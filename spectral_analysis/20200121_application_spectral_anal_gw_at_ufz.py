# import modules
import sys
import numpy as np
import pandas as pd
sys.path.append("/Users/houben/PhD/python/scripts/spectral_analysis")
# add search path for own modules on eve
sys.path.append("/home/houben/python/scripts/spectral_analysis")
# own modules
from power_spectrum import power_spectrum
from plot_power_spectra import plot_spectrum
from shh_analytical import shh_analytical_fit, shh_analytical
# add search path for own modules

# Load the data
birkach = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/birkach_quadratic.txt"
stegaurach = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/stegaurach_quadratic.txt"
strullendorf_west = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/strull_west_quadratic.txt"
strullendorf_nord = "/Users/houben/phd/application_spectral_analysis/main/gwms/data/strull_nord_quadratic.txt"
xs = [25800,600,7000,5100]
Ls = [35000,3000,8000,7800]
names = [birkach, stegaurach, strullendorf_west, strullendorf_nord]
name = birkach
x = 4400
L = 4500
m = None
n = None
norm = False
convergence = 0.01
time_step_size = 86400
cut_freq = None
combine_df = False


# Load the data and take right part of it
head_df = pd.read_csv(
    name,
    sep=" ",
    header=None,
    names=["date", "head"],
)
head_df["date"] = pd.to_datetime(head_df["date"])

# load recharge data
recharge_df = pd.read_csv(
    "/Users/houben/phd/mHM/GW_recharge_main/4348000.0_5454000.0_recharge.txt",
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
    # convert mm/d to recharge along the aquifer in m2/s
    recharge_time_series = [i / 86400 / 1000 * L for i in recharge_time_series]
    head_time_series = combined_df["head"].tolist()
else:
    recharge_time_series = recharge_df["recharge"].tolist()
    head_time_series = head_df["head"].tolist()

'''
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
plt.plot(recharge_time_series)
plt.show()
recharge_time_series = recharge_time_series[begin_index:end_index]
head_time_series = head_time_series[begin_index+shift:end_index+shift]
####################################################################
'''

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

# cut the spectra
# cut_value of frequency
#cut_array = np.less(frequency, cut_freq)
#Sww = Sww[cut_array]
#Shh = Shh[cut_array]
#frequency = frequency[cut_array]


# fit the power spectrum with the analytical solution
try:
    popt, pcov = shh_analytical_fit(
        Sww=Sww,
        Shh=Shh,
        f=frequency,
        x=x,
        m=m,
        n=n,
        L=L,
        norm=False,
        convergence=convergence,
    )
except RuntimeError:
    print("Optimal parameters not found...")
    popt, pcov = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]
    print("popt and pcov have been set to np.nan")
except ValueError:
    print("either ydata or xdata contain NaNs, or if incompatible options are used")
    popt, pcov = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]
except OptimizeWarning:
    print("Covariance of the parameters could not be estimated.")
    #popt, pcov = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]

Sy = abs(popt[0])
T = abs(popt[1])

# calculate the fitted power spectra
Shh_fitted = shh_analytical(
    (frequency, Sww),
    Sy,
    T,
    x,
    L,
    m=n,
    n=m,
    norm=norm,
)

data = np.vstack((Shh, Shh_fitted))

labels = [
    "Shh numerical",
    "Shh fitted"
]
linestyle = ["-", "-"]
marker = ["", "d"]
figtxt ="Derived Parameter:    S = %1.3e, T = %1.3e" % (
    Sy,
    T,
)

# plot only spectrum of Shh
plot_spectrum(
    [Shh],
    frequency_output,
    heading="head spectrum",
    labels=["power_spec_out"],
    path="/Users/houben/phd/application_spectral_analysis/main/gwms/data",
    linestyle=["-"],
    marker=[""],
    lims=[(2e-9,7e-6),(1e-5,1e7)],
)

# plot only spectrum of Sww
plot_spectrum(
    [Sww],
    frequency_input,
    heading="recharge spectrum",
    labels=["power_spec_out"],
    path="/Users/houben/phd/application_spectral_analysis/main/gwms/data",
    linestyle=["-"],
    marker=[""],
    lims=[(2e-9,7e-6),(1e-10,1e4)],
)

plot_spectrum(
    data,
    frequency,
    heading="TEST",
    labels=["power_spec_out", "analytical fit"],
    path="/Users/houben/phd/application_spectral_analysis/main/gwms/data",
    linestyle=["-", " "],
    marker=["", "*"],
    figtxt=figtxt,
    lims=[(2e-9,7e-6),(1e-5,1e7)],
)




# get the characteristic time
from calc_tc import calc_tc
tc = calc_tc(L,Sy,T,which="dupuit")
# define a (discharge constant)
#a = np.pi ** 2 * T / (4 * L ** 2)
# define tc (characteristic time scale)
#tc = Sy / a
