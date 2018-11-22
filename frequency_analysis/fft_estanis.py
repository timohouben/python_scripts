# =============================================================================
# import modules
# =============================================================================

# import sys and set path to the module
import sys
sys.path.append("/Users/houben/PhD/python/scripts/head_ogs_vs_gw-model/transient")
import scipy.fftpack as scpfft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import scipy.optimize as optimization
import textwrap as tw

fft_data =  np.loadtxt("/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30/testing2/Groundwater@UFZ_eve_HOMO_276_D_4_results/head_ogs_obs_0400_mean.txt")
recharge = np.loadtxt("/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30/testing2/Groundwater@UFZ_eve_HOMO_276_D_4_results/rfd_curve#1.txt")

def fft_psd(fft_data=fft_data,
            recharge=recharge,
            aquifer_thickness=30,
            aquifer_length=1000,
            distance_to_river=1000,
            path_to_project='/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30/testing2/Groundwater@UFZ_eve_HOMO_276_D_4_results',
            single_file='/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30/testing2/Groundwater@UFZ_eve_HOMO_276_D_4_results/transect_01_ply_obs_0400_t6_GROUNDWATER_FLOW.tec',
            method='scipyfft',
            fit=True, savefig=True,
            saveoutput=True,
            a_l=None, t_l=None,
            weights_l=[1,1,1,1,1],
            o_i='oi',
            time_step_size=86400,
            windows=None,
            obs_point='no_obs_given',
            threshold=1e-6):

    # =============================================================================
    # initialize the file for output
    # =============================================================================
    with open(str(path_to_project) + 'PSD_output.txt', 'a') as file:
        file.write('date time method T_l[m2/s] kf_l[m/s] Ss_l[1/m] D_l[m2/s] a_l t_l[s] path_to_project observation_point\n')
    file.close()

    # check if recharge and fft_data have the same length and erase values in the end
    if len(recharge) > len(fft_data):
        print('Your input and output data have a different length. Adjusting to the smaller one by deleting last entries.')
        recharge = recharge[:len(fft_data)]
    elif len(recharge) < len(fft_data):
        print('Your input and output data have a different length. Adjusting to the smaller one by deleting last entries.')
        fft_data = fft_data[:len(recharge)]



    # define the sampling frequency/time step
    # -------------------------------------------------------------------------
    sampling_frequency = 1./time_step_size    # [Hz] second: 1, day: 1.1574074074074E-5

    # detrend input and output signal
    # -------------------------------------------------------------------------
    #recharge_detrend = signal.detrend(recharge, type='linear')
    #fft_data_detrend = signal.detrend(fft_data, type='linear')
    recharge_detrend = recharge
    fft_data_detrend = fft_data

    # different methodologies for power spectral density
    # -------------------------------------------------------------------------


    if method == 'scipyfft':
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        global power_spectrum_input
        power_spectrum_input = (abs(scpfft.fft(recharge_detrend)[:len(fft_data_detrend)/2])**2)[1:]
        global power_spectrum_output
        power_spectrum_output = (abs(scpfft.fft(fft_data_detrend)[:len(fft_data_detrend)/2])**2)[1:]
        global power_spectrum_result
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (abs(scpfft.fftfreq(len(fft_data_detrend), time_step_size))[:len(fft_data_detrend)/2])[1:]
        if o_i == 'i':
            power_spectrum_result = power_spectrum_input
        elif o_i == 'o':
            power_spectrum_result = power_spectrum_output

        '''
        # delete values with smal frequencies
        i = 0
        for i, value in enumerate(frequency_input):
            if value > threshold:
                for j in range(i,len(frequency_input)):
                    frequency_input = np.delete(frequency_input, i)
                    power_spectrum_result = np.delete(power_spectrum_result, i)
         '''


    if method == 'pyplotwelch':
        # =========================================================================
        # method x: Pyplot PSD by Welch
        #           https://matplotlib.org/api/_as_gen/matplotlib.pyplot.psd.html
        # =========================================================================
        power_spectrum_input, frequency_input = plt.psd(recharge_detrend,
                                                        Fs=sampling_frequency)
        power_spectrum_output, frequency_output = plt.psd(fft_data_detrend,
                                                        Fs=sampling_frequency)
        # delete first value (which is 0) because it makes trouble with fitting
        frequency_output = frequency_output[1:]
        frequency_input = frequency_input[1:]
        power_spectrum_result = (power_spectrum_output / power_spectrum_input)[1:]
        if o_i == 'i':
            power_spectrum_result = power_spectrum_input[1:]
        elif o_i == 'o':
            power_spectrum_result = power_spectrum_output[1:]

        '''
        # delete values with smal frequencies
        i = 0
        for i, value in enumerate(frequency_input):
            if value > threshold:
                for j in range(i,len(frequency_input)):
                    frequency_input = np.delete(frequency_input, i)
                    power_spectrum_result = np.delete(power_spectrum_result, i)
        '''




    # plot the resulting power spectrum
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1/s")
    #ax.set_ylim(1e-3,1e6)
    #ax.plot(freq_month[ind],psd)
    ax.plot(frequency_input, power_spectrum_result, label='PSD')
    ax.set_title('power spectrum density for observation point ' + str(obs_point)
    + '\n' + 'method: ' + str(method))
    #ax.set_title('power spectruml density for recharge')
    ax.grid(color='grey', linestyle='--', linewidth=0.5, which='both')



    # =========================================================================
    # Fit the power spectrum
    # =========================================================================

    if fit == True:

        # employ a filter on the spectrum to optimize the fit
        # ---------------------------------------------------------------------
        # define wondow size based on amount of values (1/100 -> 100 windows)
        if windows == None:
            windows = len(power_spectrum_result)/10

        else:
            print('Data Points in PSD: ' + str(len(power_spectrum_result)))
        window_size = np.around((len(power_spectrum_result)/windows),0)
        if window_size % 2 == 0:
            window_size = window_size + 1
        elif window_size < 2:
            window_size = 2
        power_spectrum_result_filtered = signal.savgol_filter(power_spectrum_result, window_size, 2)
        ax.plot(frequency_input, power_spectrum_result_filtered, label='filtered PSD')


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
            # weights = [1,1,1] # give the weights for each segment, amount of values specifies the amount of segments
            data_per_segment = len(power_spectrum_result_filtered) / len(weights_l)
            for weight_l in weights_l:
                sigma_l = np.append(sigma_l,np.full(data_per_segment,weight_l))
            if len(power_spectrum_result_filtered) % len(weights_l) != 0:
                for residual in range(len(power_spectrum_result_filtered) % len(weights_l)):
                    sigma_l = np.append(sigma_l,weights_l[-1])

            # define the function to fit (linear aquifer model):
            def linear_fit(w_l, a_l, t_l):
                return (1. / (a_l**2 * ( 1 + ((t_l**2) * (w_l**2)))))              # method 1
                #return (1. / (a_l * ( 1 + ((t_l**2) * (w_l**2)))))                 # method 2
                #return (1. / (a_l**2 * ( 1 + ((t_l**2) * ((w_l/2./np.pi)**2)))))   # method 3
                #return (1. / (a_l * ( 1 + ((t_l**2) * ((w_l/2./np.pi)**2)))))      # method 4
            # perform the fit
            popt_l, pcov_l = optimization.curve_fit(linear_fit,
                                              frequency_input,
                                              power_spectrum_result_filtered,
                                              p0=initial_guess,
                                              sigma=sigma_l)
# changed optimization from filterered data
            # abs to avoid negative values from optimization
            t_l = abs(popt_l[1])
            a_l = abs(popt_l[0])
            t_l = t_l

        # Plot the linear fit model
        # ---------------------------------------------------------------------
        linear_model = []
        # fitting model for the linear reservoir (Gelhar, 1993)
        for i in range(0,len(frequency_input)):
            line = 1. / (a_l**2 * ( 1 + ((t_l**2) * (frequency_input[i]**2))))               # method 1
            #line = 1 / (a_l * ( 1 + ((t_l**2) * (frequency_input[i]**2))))                  # method 2
            #line = 1 / (a_l**2 * ( 1 + ((t_l**2) * ((frequency_input[i]/2./np.pi)**2))))    # method 3
            #line = 1 / (a_l * ( 1 + ((t_l**2) * ((frequency_input[i]/2./np.pi)**2))))       # method 4
            linear_model.append(line)
        ax.plot(frequency_input, linear_model, label='linear model')

        # calculate aquifer parameters
        # ---------------------------------------------------------------------
        print("Multiplication of t_l for D_l? 86400? or none?")
        T_l = a_l * aquifer_length**2 / 3.
        kf_l = T_l / aquifer_thickness
        S_l = a_l * t_l
        Ss_l = S_l / aquifer_thickness
        D_l = aquifer_length**2 / (3 * t_l)
        #D_l = aquifer_length**2 * 4 / (np.pi**2 * t_l)
        output_l = ('Linear model:\n ' +
              'T [m2/s]: ' + '%0.4e' % T_l  + '\n  ' +
              'Ss [1/m]: ' + '%0.4e' % Ss_l + '\n  ' +
              'kf [m/s]: ' + '%0.4e' % kf_l + '\n  ' +
              'D [m2/s]: ' + '%0.4e' % D_l + '\n  ' +
              'a : ' + '%0.4e' % a_l+ '\n  ' +
              't_c [s]: ' + '%0.4e' % t_l
              )
        print(output_l)
        fig_txt = tw.fill(output_l, width=200)


        #annotate the figure
        #fig_txt = tw.fill(tw.dedent(output), width=120)
        plt.figtext(0.5, 0.2, fig_txt, horizontalalignment='center',
                    bbox=dict(boxstyle="round", facecolor='#F2F3F4',
                              ec="0.5", pad=0.5, alpha=1))

    plt.legend(loc='best')
    plt.show()
    if savefig == True:
        fig.savefig(str(path_to_project) + 'PSD_' + str(method) + '_' +
                    str(os.path.basename(str(path_to_project)[:-1])) + '_' +
                    str(obs_point) + ".png")
        plt.close()

    if fit == True and saveoutput == True:
        with open(str(path_to_project) + 'PSD_output.txt', 'a') as file:
            file.write(str(datetime.datetime.now()) + ' ' + method + ' ' +
                                str(T_l) + ' ' + str(kf_l) + ' ' +
                                str(Ss_l) + ' ' + str(D_l) + ' ' +
                                str(a_l) + ' ' + str(t_l) + ' ' +
                                str(path_to_project) + str(single_file) +
                                ' ' + str(obs_point) + '\n')
        file.close()
    return T_l, kf_l, Ss_l, D_l, a_l, t_l


fft_psd(fft_data=fft_data, recharge=recharge, method='scipyfft', obs_point='obs_0400')
