# import modules
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from Functions import alpha_reg, filter_Eck
from transfer_functions import discharge_ftf_fit, discharge_ftf

path = "/Users/houben/phd/application_spectral_analysis/main/pegel"
path_d = "/Users/houben/phd/application_spectral_analysis/main/pegel/Ganglinien_pegel/kemmern_quadratic.txt"
path_r = "/Users/houben/phd/application_spectral_analysis/main/recharge/4416000.0_5526000.0_stegaurach_recharge.txt"
aquifer_length = 1000
time_step_size = 86400
area = 4243.20082734

# load the raw data
discharge_df = pd.read_csv(
    path_d,
    sep=" ",
    header=None,
    names=["date", "discharge"],
)

discharge_df["date"] = pd.to_datetime(discharge_df["date"])

recharge_df = pd.read_csv(
    path_r,
    sep=" ",
    header=None,
    names=["date", "recharge"],
)
recharge_df["date"] = pd.to_datetime(recharge_df["date"])

# combine dataframes and remove rows with nans
combined_df = pd.merge_ordered(recharge_df, discharge_df, how="inner")
date_min = combined_df["date"].min()
date_max = combined_df["date"].max()
period = combined_df["date"].max() - combined_df["date"].min()
print("Start/end/length of series where head measurements and recharge overlap: " + str(date_min) + "/" + str(date_max) + "/" + str(period))
recharge_time_series = combined_df["recharge"].tolist()
discharge_time_series = combined_df["discharge"].tolist()

# slice both time series
cut_index = 8000
recharge_time_series = recharge_time_series[cut_index:]
discharge_time_series = discharge_time_series[cut_index:]

# multiply recharge by area
recharge_time_series = [i*area for i in recharge_time_series]


BFImax = 0.5

alpha = alpha_reg(discharge_time_series, area)
baseflow_time_series = filter_Eck(BFImax, discharge_time_series, alpha)


#plt.plot(discharge_time_series)
#plt.plot(baseflow_time_series)


popt, pcov, frequency_input, power_spectrum_result = discharge_ftf_fit(recharge_time_series, baseflow_time_series, time_step_size, 1000, method='scipyffthalf', initial_guess=1e-2)
power_spectrum_input = power_spectrum(recharge_time_series, power_spectrum_result, time_step_size, method="scipyffthalf", o_i="i")[1]


plt.loglog(frequency_input, power_spectrum_result)
plt.loglog(frequency_input, power_spectrum_input)

#data = np.hstack((spectrum_input, power_spectrum_result))
#plot_spectrum(data, frequency_input, path=path, name="test")

'''
def plot_spectrum(
    data,
    frequency,
    name=None,
    labels=None,
    path=None,
    lims=None,
    linestyle="-",
    marker="",
    #markersize=None,
    grid="both",
    unit="[Hz]",
    heading="None",
    figtxt=None,
    comment="",
):
'''
