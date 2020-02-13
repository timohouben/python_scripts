"""
Script to perform a small sensitivity analysis for a fit of the shh_anlytical
solution to 4 example groundwater head time series. The target parameters are
T, S and R^2 of the fit. The variables parameters are L and x.
Constraints: L must not be bigger than x!

L and x are drawn randomly from a gaussian?? distribution.
"""
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

#############################################
# Set parameters for script
# comment your analysis
comment = "1_"
# path to save images
save_path = "/Users/houben/phd/application_spectral_analysis/main/plots/spectral_analysis/sensitivity_analysis"
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
names_h = ["Birkach", "Stegaurach", "Strullendorf_West", "Strullendorf_Nord"]
names_r = ["Birkach", "Stegaurach", "Strullendorf_West", "Strullendorf_Nord"]
paths_h = [birkach_h, stegaurach_h, strullendorf_west_h, strullendorf_nord_h]
paths_r = [birkach_r, stegaurach_r, strullendorf_west_r, strullendorf_nord_r]
# Further parameters for the fit.
m = None
n = None
norm = False
convergence = 0.01
time_step_size = 86400
# cut higher frequencies than cut_freq_higher
cut_freq_higher = 1e-4
# cut lower frequencies than cut_freq_lower
cut_freq_lower = 1e-9
combine_df = False
detrend_ts = True
#############################################

#############################################
# Scripts begins
#############################################
# Draw samples of L and x and pop if L < x
np.random.seed(1340)
epsilon = 5
# maximum value of L
threshold = 40000
Ls = []
xs = []


def get_samples(a, b, c, which="uni"):
    if which == "uni":
        samples = np.random.uniform(a, b, c)
    elif which == "lognorm":
        samples = np.random.lognormal(a, b, c)
    else:
        print("Which mus be either 'uni' or 'log'.")
    return samples


def erase_x_greater_L(Ls_sample, xs_sample):
    Ls = []
    xs = []
    for L, x in zip(Ls_sample, xs_sample):
        if (L > x) & (L < 40000):
            Ls.append(L)
            xs.append(x)
    return Ls, xs


def normalize_by_mean(series):
    series_mean = np.mean(series)
    series_norm = [i / series_mean for i in series]
    return list(series_norm)


#Ls_sample = get_samples(10, 40000, 200)
#xs_sample = get_samples(10, 40000, 200)
Ls_sample = get_samples(10, 4, 500, "lognorm")
xs_sample = get_samples(10, 4, 500, "lognorm")
Ls, xs = erase_x_greater_L(Ls_sample, xs_sample)


"""
# loop to get a uniform distribution even after popping values, does not yet work properly, Popping of wrong values missing
count_L = plt.hist(Ls, bins=20, alpha=0.5, label="aquifer length")[0]
count_x = plt.hist(xs, bins=20, alpha=0.5, label="position")[0]
plt.close()
count_L_norm = normalize_by_mean(count_L)
count_x_norm = normalize_by_mean(count_x)
while (all(abs(i) > epsilon for i in count_L_norm) is False) & (all(abs(i) > epsilon for i in count_x_norm) is False):
    print(count_L_norm)
    print(count_x_norm)
    print("No... New iteration")
    Ls_sample = get_samples(10, 40000, 200)
    xs_sample = get_samples(10, 40000, 200)
    Ls_temp, xs_temp = erase_x_greater_L(Ls_sample, xs_sample)
    Ls = Ls_temp + Ls
    xs = xs_temp + xs
    count_L = plt.hist(Ls, bins=20, alpha=0.5, label="aquifer length")[0]
    count_x = plt.hist(xs, bins=20, alpha=0.5, label="position")[0]
    count_L_norm = normalize_by_mean(count_L)
    count_x_norm = normalize_by_mean(count_x)
else:
    print("FINISHED!!!")

"""

L_and_x_input = [(L, x) for L, x in zip(Ls, xs)]
L_and_x_input.sort()
np.savetxt(save_path + "/" + comment + "L_and_x_input.txt", L_and_x_input)
# Scatter plot
plt.scatter(x = [i[0] for i in L_and_x_input], y = [i[1] for i in L_and_x_input])
plt.yscale("log")
plt.xscale("log")
plt.ylabel("x")
plt.xlabel("L")
plt.ylim((100,50000))
plt.xlim((100,50000))
plt.title("Input parameter for L and x")
plt.savefig(
    save_path + "/" + comment + "L_and_x_input_scatter.png", bbox_inches="tight", dpi=300
)
plt.close()
# line plot
L_plot, x_plot = plt.plot(L_and_x_input)
plt.ylabel("[m]")
plt.xlabel("count")
plt.title("Input parameter for L and x")
plt.legend([L_plot, x_plot], ["aquifer length", "position"], loc=1)
plt.savefig(
    save_path + "/" + comment + "L_and_x_input_plot.png", bbox_inches="tight", dpi=300
)
plt.close()
# histogram
plt.hist(Ls, bins=20, alpha=0.5, label="aquifer length")
plt.hist(xs, bins=20, alpha=0.5, label="position")
plt.ylabel("count")
plt.xlabel("value")
plt.title("Histograms of L and x input parameter")
plt.legend()
plt.savefig(
    save_path + "/" + comment + "L_and_x_input_hist.png", bbox_inches="tight", dpi=300
)
plt.close()

# initialize a dataframe to save the results to
pd.set_option("precision", 10)
columns = [
    "name_h",
    "name_r",
    "L",
    "x",
    "S_out",
    "T_out",
    "tc_out",
    "cov",
    "time_step_size",
    "cut_freq_higher",
    "cut_freq_lower",
]
results = pd.DataFrame(columns=columns)

# loop over all paths and names with different aquifer lengths and positions
for name_h, path_h, name_r, path_r in zip(names_h, paths_h, names_r, paths_r):
    for L, x in zip(Ls, xs):
        print("Currently fitting: " + str(name_h) + " ...")
        # Load the data
        head_df = pd.read_csv(path_h, sep=" ", header=None, names=["date", "head"])
        head_df["date"] = pd.to_datetime(head_df["date"])

        # load recharge data
        recharge_df = pd.read_csv(
            path_r, sep=" ", header=None, names=["date", "recharge"]
        )
        recharge_df["date"] = pd.to_datetime(recharge_df["date"])

        if combine_df is True:
            # combine dataframes and remove rows with nans
            combined_df = pd.merge_ordered(recharge_df, head_df, how="inner")
            date_min = combined_df["date"].min()
            date_max = combined_df["date"].max()
            period = combined_df["date"].max() - combined_df["date"].min()
            print(
                "Start/end/length of series where head measurements and recharge overlap: "
                + str(date_min)
                + "/"
                + str(date_max)
                + "/"
                + str(period)
            )
            recharge_time_series = combined_df["recharge"].tolist()
            head_time_series = combined_df["head"].tolist()
        else:
            recharge_time_series = recharge_df["recharge"].tolist()
            head_time_series = head_df["head"].tolist()
            # modify the time series so that both have same length
            # assume: recharge is longest
            recharge_time_series = recharge_time_series[-len(head_time_series) :]

        if detrend_ts is True:
            head_time_series = detrend(head_time_series)
            recharge_time_series = detrend(recharge_time_series)
        else:
            pass

        # convert mm/d to recharge along the aquifer in m2/s
        recharge_time_series = [i / 86400 / 1000 for i in recharge_time_series]

        """
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
        #recharge_time_series = np.loadtxt("/Users/houben/phd/modelling/20190717_SA_hetero_block_2/1076_border_50_stor_0.0001_rech_mHM/rfd_curve#1_y_values.txt")
        #head_time_series = np.loadtxt("/Users/houben/phd/modelling/20190717_SA_hetero_block_2/1076_border_50_stor_0.0001_rech_mHM/head_ogs_obs_00500_mean.txt")
        #x = 500
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
        end_index = 10000
        shift = 0
        #%matplotlib qt
        import matplotlib.pyplot as plt
        #plt.plot(recharge_time_series)
        #plt.show()
        recharge_time_series = recharge_time_series[begin_index:end_index]
        head_time_series = head_time_series[begin_index+shift:end_index+shift]
        ####################################################################
        """

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

        # cut higher frequencies than cut_freq_higher
        cut_array_higher = np.less(frequency_input, cut_freq_higher)
        Sww = Sww[cut_array_higher]
        Shh = Shh[cut_array_higher]
        frequency_input = frequency_input[cut_array_higher]
        frequency_output = frequency_output[cut_array_higher]
        # cut lower frequencies than cut_freq_lower
        cut_array_lower = np.invert(np.less(frequency_input, cut_freq_lower))
        Sww = Sww[cut_array_lower]
        Shh = Shh[cut_array_lower]
        frequency_input = frequency_input[cut_array_lower]
        frequency_output = frequency_output[cut_array_lower]

        # fit the power spectrum with the analytical solution
        try:
            popt, pcov = shh_analytical_fit(
                Sww=Sww,
                Shh=Shh,
                f=frequency_input,
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
            print(
                "either ydata or xdata contain NaNs, or if incompatible options are used"
            )
            popt, pcov = [np.nan, np.nan], [[np.nan, np.nan], [np.nan, np.nan]]
        except OptimizeWarning:
            print("Covariance of the parameters could not be estimated.")
            # popt, pcov = [np.nan, np.nan], [[np.nan, np.nan],[np.nan, np.nan]]

        Sy = abs(popt[0])
        T = abs(popt[1])

        # calculate the fitted power spectra
        Shh_fitted = shh_analytical(
            (frequency_input, Sww), Sy, T, x, L, m=n, n=m, norm=norm
        )

        # get the characteristic time
        tc = calc_tc(L, Sy, T, which="dupuit")
        # define a (discharge constant)
        # a = np.pi ** 2 * T / (4 * L ** 2)
        # define tc (characteristic time scale)
        # tc = Sy / a

        data = np.vstack((Shh, Shh_fitted))

        labels = ["Shh numerical", "Shh fitted"]
        linestyle = ["-", "-"]
        marker = ["", "d"]
        figtxt = (
            "Derived Parameter:    S = %1.3e, T = %1.3e [m2/s], tc = %1.3e [d]\nInput Parameter:        L = %0.0f, x = %0.0f"
            % (Sy, T, tc, L, x)
        )
        print("Currently plotting: " + str(name_h) + " ...")
        # plot only spectrum of Shh
        plot_spectrum(
            [Shh],
            frequency_output,
            heading="Shh - Head Power Spectrum " + name_h,
            labels=["Shh obs"],
            path=save_path,
            linestyle=["-"],
            marker=[""],
            lims=[(2e-9, 7e-6), (1e-5, 1e7)],
            name="Shh_" + name_h,
        )

        # plot only spectrum of Sww
        plot_spectrum(
            [Sww],
            frequency_input,
            heading="Sww - Recharge Power Spectrum  " + name_r,
            labels=["Sww mHM"],
            path=save_path,
            linestyle=["-"],
            marker=[""],
            lims=[(2e-9, 7e-6), (1e-20, 1e-9)],
            name="Sww_" + name_r,
        )

        # plot Shh and the fitted spectrum
        plot_spectrum(
            data,
            frequency_input,
            heading="Shh - Head Power Spectrum " + name_h,
            labels=["Shh obs", "Shh fit"],
            path=save_path,
            linestyle=["-", " "],
            marker=["", "*"],
            figtxt=figtxt,
            lims=[(2e-9, 7e-6), (1e-7, 1e8)],
            name="Shh_fitted_"
            + name_h
            + "_R_"
            + name_r
            + "_L_"
            + str(L)
            + "_x_"
            + str(x),
        )

        results_temp = {
            "name_h": name_h,
            "name_r": name_r,
            "L": L,
            "x": x,
            "S_out": Sy,
            "T_out": T,
            "tc_out": tc,
            "cov": pcov,
            "time_step_size": time_step_size,
            "cut_freq_higher": cut_freq_higher,
            "cut_freq_lower": cut_freq_lower,
        }
        results = results.append(other=results_temp, ignore_index=True, sort=False)

# set path to results incl file name of results
path_to_results_df = save_path + "/" + comment + "results.csv"
results.to_csv(path_to_results_df)
