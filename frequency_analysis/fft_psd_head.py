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
from conf_head_ogs_vs_gw_model_trans import gethead_ogs_each_obs, getrecharge, gethead_gw_model_each_obs, make_array_gw_model, split_gw_model, getlist_gw_model, convert_obs_list_to_index
from ogs5py.reader import readtec_polyline
import scipy.fftpack as scpfft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.optimize as optimization
import textwrap as tw

# =============================================================================
# global variables set manually
# =============================================================================
which_data_to_plot = 1 # 1: ogs, 2: gw_model 3: recharge
path_to_project = "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_HOMO_276_D/"
name_of_project_gw_model = "sinus"
name_of_project_ogs = "transect_01"
process = 'GROUNDWATER_FLOW'
which = 'max'       # min, max, mean
time_steps = 8400    # this is the value which is given in the ogs input file .tim. It will result in a total of 101 times because the initial time is added.
# variables for FFT
obs_point = ''

# =============================================================================
# global variables set automatically
# =============================================================================
print ("Reading .tec-files...")
tecs = readtec_polyline(task_id=name_of_project_ogs,task_root=path_to_project, single_file="/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_HOMO_276_D/transect_01_ply_obs_0200_t21_GROUNDWATER_FLOW.tec")
print ("Done.")

# =============================================================================
# get data dependent on which_data_to_plot
# =============================================================================
if which_data_to_plot == 1:
    head_ogs = gethead_ogs_each_obs(process, obs_point, which, time_steps, tecs=tecs, single_file=True)
    fft_data = head_ogs
    recharge = getrecharge(path_to_project=path_to_project, name_of_project_ogs=name_of_project_ogs, time_steps=time_steps)
elif which_data_to_plot == 2:
    head_gw_model = gethead_gw_model_each_obs(make_array_gw_model(
                        split_gw_model(getlist_gw_model(str(path_to_project) 
                        + str(name_of_project_gw_model) 
                        + '/H.OUT'), index=2)), convert_obs_list_to_index('obs_0990'))
    fft_data = head_gw_model
    recharge = getrecharge(path_to_project=path_to_project, name_of_project_ogs=name_of_project_ogs, time_steps=time_steps)
elif which_data_to_plot == 3:
    recharge = getrecharge(path_to_project=path_to_project, name_of_project_ogs=name_of_project_ogs, time_steps=time_steps)
    fft_data = recharge
 
# convert recharge from list to array    
recharge = np.asarray([float(i) for i in recharge])

# =============================================================================
# Calculate the discrete fourier transformation    
# =============================================================================

def fft_psd(method='fourier', fit=False, savefig=False, a_l=None, t_l=None, a_d=None, t_d=None, fft_data=fft_data, weights_l=[1,1,1], weights_d=[1,1,1], o_i='oi',aquifer_thickness = 30, aquifer_length = 1000, distance_to_river = 800, windows=None):
     
    # define the sampling frequency/time step
    # -------------------------------------------------------------------------
    time_step_size = 86400  # [s]
    sampling_frequency = 1./time_step_size    # [Hz] second: 1, day: 1.1574074074074E-5
     
    # detrend input and output signal
    # -------------------------------------------------------------------------
    recharge_detrend = signal.detrend(recharge, type='linear')
    fft_data_detrend = signal.detrend(fft_data, type='linear')    
    
    # different methodologies for power spectral density
    # -------------------------------------------------------------------------

    if method == 'scipyfftnormt':
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2 / T
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = ((abs(scpfft.fft(recharge_detrend)[:len(fft_data_detrend)/2])**2) / (len(fft_data)*time_step_size))[1:]
        power_spectrum_output = ((abs(scpfft.fft(fft_data_detrend)[:len(fft_data_detrend)/2])**2) / (len(fft_data)*time_step_size))[1:]
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (abs(scpfft.fftfreq(len(fft_data_detrend), time_step_size))[:len(fft_data_detrend)/2])[1:]
        if o_i == 'i':
            power_spectrum_result = power_spectrum_input
        elif o_i == 'o':
            power_spectrum_result = power_spectrum_output

    if method == 'scipyfftnormn':
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2 / N
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = ((abs(scpfft.fft(recharge_detrend)[:len(fft_data_detrend)/2])**2) / len(fft_data))[1:]
        power_spectrum_output = ((abs(scpfft.fft(fft_data_detrend)[:len(fft_data_detrend)/2])**2) / len(fft_data))[1:]
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (abs(scpfft.fftfreq(len(fft_data_detrend), time_step_size))[:len(fft_data_detrend)/2])[1:]
        if o_i == 'i':
            power_spectrum_result = power_spectrum_input
        elif o_i == 'o':
            power_spectrum_result = power_spectrum_output

    if method == 'scipyfft':
        # =========================================================================
        # method x: Periodogram: Power Spectral Density: abs(X(w))^2
        #           http://staff.utia.cas.cz/barunik/files/QFII/04%20-%20Seminar/04-qf.html
        # =========================================================================
        power_spectrum_input = (abs(scpfft.fft(recharge_detrend)[:len(fft_data_detrend)/2])**2)[1:]
        power_spectrum_output = (abs(scpfft.fft(fft_data_detrend)[:len(fft_data_detrend)/2])**2)[1:]
        power_spectrum_result = power_spectrum_output / power_spectrum_input
        frequency_input = (abs(scpfft.fftfreq(len(fft_data_detrend), time_step_size))[:len(fft_data_detrend)/2])[1:]
        if o_i == 'i':
            power_spectrum_result = power_spectrum_input
        elif o_i == 'o':
            power_spectrum_result = power_spectrum_output            

    elif method == 'scipywelch':    
        # =========================================================================
        # method x: scipy.signal.welch
        #           https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html#r145
        # =========================================================================
        frequency_input, power_spectrum_input = signal.welch(recharge_detrend, 
                                                            sampling_frequency, 
                                                            nperseg=16000, 
                                                            window='hamming')
        frequency_output, power_spectrum_output = signal.welch(fft_data_detrend, 
                                                              sampling_frequency, 
                                                              nperseg=16000, 
                                                              window='hamming')
        frequency_output = frequency_output[1:]
        frequency_input = frequency_input[1:]
        power_spectrum_result = (abs((power_spectrum_output / power_spectrum_input))**2)[1:]
        if o_i == 'i':
            power_spectrum_result = power_spectrum_input[1:]
        elif o_i == 'o':
            power_spectrum_result = power_spectrum_output[1:]     

    elif method == 'pyplotwelch':    
        # =========================================================================
        # method x: Pyplot PSD by Welch
        #           https://matplotlib.org/api/_as_gen/matplotlib.pyplot.psd.html
        # =========================================================================       
        power_spectrum_input, frequency_input = plt.psd(recharge_detrend,
                                                        Fs=sampling_frequency)
        power_spectrum_output, frequency_output = plt.psd(fft_data_detrend,
                                                        Fs=sampling_frequency)
        frequency_output = frequency_output[1:]
        frequency_input = frequency_input[1:]
        power_spectrum_result = (power_spectrum_output / power_spectrum_input)[1:]
        if o_i == 'i':
            power_spectrum_result = power_spectrum_input[1:]
        elif o_i == 'o':
            power_spectrum_result = power_spectrum_output[1:]    
         
         PYPLOTWELCH SOLLTE NUN FUNKTIONIEREN; DANN HIER WEITER MACHEN!   
            
    elif method == 'scipyperio':    
        # =========================================================================
        # method x: Scipy.signal.periodogram
        #           https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.signal.periodogram.html
        # =========================================================================     
        frequency_input, power_spectrum_input = signal.periodogram(recharge_detrend,
                                                        fs=sampling_frequency)
        frequency_output, power_spectrum_output = signal.periodogram(fft_data_detrend,
                                                        fs=sampling_frequency)
        power_spectrum_result = power_spectrum_output
        if o_i == True:
            power_spectrum_result = power_spectrum_output / power_spectrum_input          
        
    elif method == 'spectrumperio':    
        # =========================================================================
        # method x: Spectrum.periodogram
        #           http://thomas-cokelaer.info/software/spectrum/html/user/ref_fourier.html#spectrum.periodogram.Periodogram
        # =========================================================================              
        from spectrum import WelchPeriodogram
        power_spectrum_input, empty = WelchPeriodogram(recharge_detrend, 256)
        frequency_input = power_spectrum_input[1]
        power_spectrum_input = power_spectrum_input[0]
        power_spectrum_output, empty = WelchPeriodogram(fft_data_detrend, 256)
        frequency_output = power_spectrum_output[1]
        power_spectrum_output = power_spectrum_output[0]
        power_spectrum_result = power_spectrum_output
        if o_i == True:
            power_spectrum_result = power_spectrum_output / power_spectrum_input          
    
    '''
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
    '''          
    
    # plot the resulting power spectrum
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1,1,1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1/second")
    #ax.set_ylim(1e-3,1e6)
    #ax.plot(freq_month[ind],psd)
    ax.plot(frequency_input, power_spectrum_result, label='PSD')
    ax.set_title('power spectrum density for observation point ' + str(obs_point))
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
                return (1. / (a_l**2 * ( 1 + ((t_l**2) * (w_l**2)))))
            
            # perform the fit
            popt_l, pcov_l = optimization.curve_fit(linear_fit,
                                              frequency_input,
                                              power_spectrum_result_filtered,
                                              p0=initial_guess,
                                              sigma=sigma_l)
            a_l = popt_l[0]
            t_l = popt_l[1]

        # Plot the linear fit model
        # ---------------------------------------------------------------------
        linear_model = []
        # fitting model for the linear reservoir (Gelhar, 1993)
        for i in range(0,len(frequency_input)):
            line = 1 / (a_l**2 * ( 1 + ((t_l**2) * (frequency_input[i]**2))))
            linear_model.append(line)
        ax.plot(frequency_input, linear_model, label='linear model')
 
        # calculate aquifer parameters
        # ---------------------------------------------------------------------     
        print("Multiplication of t_l for D_l? 86400? or none?")
        T_l = a_l * aquifer_length**2 / 3.
        kf_l = T_l/30.
        Ss_l = a_l * t_l
        #D = kf / Ss
        D_l = T_l / Ss_l
        output = ('Aquifer parameters obtained by linear model\n' + 
              'T [m2/s]: ' + '%0.4e' % T_l  + '\n' +
              'Ss [1/m]: ' + '%0.4e' % Ss_l + '\n' +
              'kf [m/s]: ' + '%0.4e' % kf_l + '\n' +
              'D [m2/s]: ' + '%0.4e' % D_l + '\n' +
              'a : ' + '%0.4e' % a_l+ '\n' +
              't_c [s]: ' + '%0.4e' % t_l
              )
        print(output)
        plt.text(1e-2,4e14,output)
        
        # =====================================================================
        # Dupuit Model
        # =====================================================================
        # Step 5: least squares automatic fit for Dupuit-Aquifer model 
        # (e.g. Gelhar and Wilson, 1974): 
        # abs(H_h(w))**2 = (b/E)**2 * ( (1/O)*tanh)((1+j)*sqrt(1/2*O))*tanh((1-j)*sqrt(1/2*O))
        # O = td * w
        # E = x - x_o    distance from river
        # ---------------------------------------------------------------------
        if a_d == None and t_d == None:
            # make an initial guess for a_l, and t_l
            initial_guess = np.array([0.98e-15, 2000000])
    
            # generate a weighing array
            # ---------------------------------------------------------------------      
            # based on dividing the data into segments
            sigma_d = []
            # weights = [1,1,1] # give the weights for each segment, amount of values specifies the amount of segments
            data_per_segment = len(power_spectrum_result_filtered) / len(weights_d)
            for weight_d in weights_d:
                sigma_d = np.append(sigma_d,np.full(data_per_segment,weight_d))
            if len(power_spectrum_result_filtered) % len(weights_d) != 0:
                for residual in range(len(power_spectrum_result_filtered) % len(weights_d)):
                    sigma_d = np.append(sigma_d,weights_d[-1])
    
            # define the function to fit (linear aquifer model):
            def dupuit_fit(w_d, a_d, t_d):
                return ((1./a_d)**2 * ( (1./(t_d*w_d))*np.tanh((1+1j)*np.sqrt(1./2*t_d*w_d)).real*np.tanh((1-1j)*np.sqrt(1./2*t_d*w_d)).real))
            #((1./a_d)**2 * ( (1./(t_d*w_d))*np.tanh((1+1j)*np.sqrt(1./2*t_d*w_d))*np.tanh((1-1j)*np.sqrt(1./2*t_d*w_d)))).real
    
            
            # perform the fit
            popt_d, pcov_d = optimization.curve_fit(dupuit_fit,
                                              frequency_input,
                                              power_spectrum_result_filtered,
                                              p0=initial_guess,
                                              sigma=sigma_d)
           
            a_d = popt_d[0]
            t_d = popt_d[1]        
        
        # Plot the Dupuit model
        # ---------------------------------------------------------------------
        try:
            dupuit_model = []
            # fitting model for the linear reservoir (Gelhar, 1993)
            for i in range(0,len(frequency_input)):
                line = ((1./a_d)**2 * ( (1./(t_d*frequency_input[i]))*np.tanh((1+1j)*np.sqrt(1./2*t_d*frequency_input[i]))*np.tanh((1-1j)*np.sqrt(1./2*t_d*frequency_input[i])))).real
                dupuit_model.append(line)
            ax.plot(frequency_input, dupuit_model, label='Dupuit model')
     
            # calculate aquifer parameters
            # ---------------------------------------------------------------------       
            T_d = a_d * aquifer_thickness * distance_to_river
            kf_d = T_d/aquifer_thickness
            Ss_d = t_d * T_d / aquifer_length 
            D_d = T_d / Ss_d
            output = ('Aquifer parameters obtained by Dupuit model\n' + 
                  'T [m2/s]: ' + '%0.4e' % T_d  + '\n' +
                  'Ss [1/m]: ' + '%0.4e' % Ss_d + '\n' +
                  'kf [m/s]: ' + '%0.4e' % kf_d + '\n' +
                  'D [m2/s]: ' + '%0.4e' % D_d + '\n' +
                  'a : ' + '%0.4e' % a_d + '\n' +
                  't_c [s]: ' + '%0.4e' % t_d
                  )
            
            
            print(output)
            fig_txt = tw.fill(tw.dedent(output), width=120)
            plt.figtext(0, 0, fig_txt)
            
        except TypeError:
            print("No a_d and t_d given for Dupuit model. Automatic fit not working...")

    plt.legend(loc='best')
    plt.show()        
    if savefig == True:
        fig.savefig(str(path_to_project) + 'FREQUENCY_7_' + str(os.path.basename(str(path_to_project)[:-1])) + '_' + str(obs_point) + ".png")
    #fig.savefig(str(obs_point) + '_' + str(time.strftime("%Y%m%d%H%M%S")))
  