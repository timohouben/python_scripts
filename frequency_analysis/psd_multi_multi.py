#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 09:54:44 2018

@author: houben

This is a smal script to execute the psd_fft_head script multiple times for various model results and observation points.
"""



def psd_multi_multi(path_to_multiple_projects, aquifer_thickness, aquifer_length, comment):


    import sys
    import numpy as np

    sys.path.append("/Users/houben/PhD/python/scripts/frequency_analysis")
    sys.path.append(
        "/home/houben/python_pkg/python_scripts/python_scripts/head_ogs_vs_gw-model/transient"
    )
    from fft_psd_head import fft_psd, get_fft_data_from_simulation
    import matplotlib.pyplot as plt
    import os
    import time
    from calculate_model_params import calc_aq_param
    from get_obs import get_obs
    from own_colors import own_colors

    then = time.time()

    # methods = ['scipyperio', 'scipywelch', 'scipyffthalf']
    methods = ["scipyffthalf"]
    # methods = ['scipywelch']
    # methods = ['scipywelch', 'scipyffthalf']

    """
    # configuration for homogeneous model runs
    path_to_project_list = ["/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_1_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_2_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_3_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_4_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_5_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_6_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_7_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_8_results/"]
    single_file_names = ["transect_01_ply_obs_0100_t3_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0300_t5_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0400_t6_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0600_t8_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0800_t10_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0950_t12_GROUNDWATER_FLOW.tec"]
    obs_point_list = ["0100", "0300", "0400", "0600", "0800", "0950"]
    """
    """
    # configuration for homo/vertical/horizontal
    path_to_project_list = ["/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_HOMO_276_D/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_HORIZONTAL_276_D/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/Groundwater@UFZ_eve_VERTICAL_276_D/"]
    single_file_names = ["transect_01_ply_obs_0100_t11_GROUNDWATER_FLOW.tec",
                         "transect_01_ply_obs_0300_t31_GROUNDWATER_FLOW.tec",
                         "transect_01_ply_obs_0400_t41_GROUNDWATER_FLOW.tec",
                         "transect_01_ply_obs_0600_t61_GROUNDWATER_FLOW.tec",
                         "transect_01_ply_obs_0800_t81_GROUNDWATER_FLOW.tec",
                         "transect_01_ply_obs_0950_t96_GROUNDWATER_FLOW.tec"]
    obs_point_list = ["0100", "0300", "0400", "0600", "0800", "0950"]
    """
    """
    # configuration for homogeneous models, 2nd run
    path_to_project_list = ["/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_18_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_19_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_20_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_21_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_22_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_23_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_24_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_25_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_26_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_27_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_28_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_29_results/",
                            "/Users/houben/PhD/transect/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/Groundwater@UFZ_eve_HOMO_276_D_30_results/"]
    name_of_model_runs = ["D_18", "D_19", "D_20", "D_21", "D_22", "D_23", "D_24", "D_25", "D_26", "D_27", "D_28", "D_29", "D_30"]
    single_file_names = ["transect_01_ply_obs_0000_t1_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0200_t4_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0400_t6_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0500_t7_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0600_t8_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0800_t10_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0950_t12_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_0990_t16_GROUNDWATER_FLOW.tec",
                        "transect_01_ply_obs_1000_t17_GROUNDWATER_FLOW.tec"]
    """
    """
    ###############################################################################
    # configurations for model run: Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30/testing
    ###############################################################################

    path_to_multiple_projects = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30/testing"
    project_folder_list = [f for f in os.listdir(str(path_to_multiple_projects)) if not f.startswith(('.','fitting'))]
    project_folder_list.sort()
    aquifer_thickness = 30
    aquifer_length = 1000
    obs_point_list = ['obs_0000', 'obs_0010', 'obs_0100', 'obs_0200', 'obs_0300', 'obs_0400', 'obs_0500', 'obs_0600', 'obs_0700', 'obs_0800', 'obs_0900', 'obs_0950', 'obs_0960', 'obs_0970', 'obs_0980', 'obs_0990', 'obs_1000']
    distance_to_river_list = [1000, 990, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50, 40, 30, 20, 10, None]
    time_steps = 8401
    time_step_size = 86400
    threshold=1e-6

    ###############################################################################
    ###############################################################################
    """

    """
    ###############################################################################
    # configurations for model run: Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30/testing2
    ###############################################################################

    path_to_multiple_projects = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30/testing2"
    project_folder_list = [f for f in os.listdir(str(path_to_multiple_projects)) if not f.startswith(('.','fitting'))]
    project_folder_list.sort()
    aquifer_thickness = 30
    aquifer_length = 1000
    obs_point_list = ['obs_0000', 'obs_0200', 'obs_0400', 'obs_0500', 'obs_0600', 'obs_0800', 'obs_0950', 'obs_0990', 'obs_1000']
    distance_to_river_list = [1000, 800, 600, 500, 400, 200, 50, 10, None]
    time_steps = 8401
    time_step_size = 86400
    threshold=1e-6

    ###############################################################################
    ###############################################################################
    """

    ###############################################################################
    # configurations for model runs on EVE:
    # /work/houben/spectral_analysis/20181211_dupuit
    # has to start for each folder seperately
    ###############################################################################
    path_to_multiple_projects = path_to_multiple_projects
    project_folder_list = [
        f for f in os.listdir(str(path_to_multiple_projects)) if not f.startswith(".")
    ]
    try:
        project_folder_list.remove("fitting_results")
    except ValueError:
        pass
    project_folder_list.sort()
    obs_point_list = get_obs(path_to_multiple_projects + "/" + project_folder_list[0])[1]

    obs_locations = get_obs(
        path_to_multiple_projects + "/" + project_folder_list[0]
    )[2]

    # configuration for models on local machine
    #Ss_list = [
    #    1.20e-03,
    #    1.10e-03,
    #    1.00e-03,
    #    9.00e-04,
    #    8.00e-04,
    #    7.00e-04,
    #    6.00e-04,
    #    5.00e-04,
    #    4.00e-04,
    #    3.00e-04,
    #    2.00e-04,
    #    1.00e-04,
    #    9.00e-05,
    #]

    # configuration for models on eve

    Ss_list = [1.00E-03,
    1.00E-04,
    1.10E-03,
    1.20E-03,
    2.00E-04,
    3.00E-04,
    4.00E-04,
    5.00E-04,
    6.00E-04,
    7.00E-04,
    8.00E-04,
    9.00E-04,
    9.00E-05
    ]

    T_list = [
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
        30.00e-05,
    ]

    kf_list = [
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
    ]

    aquifer_thickness = aquifer_thickness
    S_list = [i * aquifer_thickness for i in Ss_list]

    aquifer_length = aquifer_length
    weights_d = [1, 1, 1, 1, 1]
    a_d_in = None
    t_d_in = None
    # a_d_in=3e-8
    # t_d_in=6.8e+7
    time_steps = 8401
    time_step_size = 86400
    comment = comment
    threshold = 1
    fit = True
    mean_thick = False
    icsub = None
    target = False
    cutoff = None
    # config for shh/sww
    #ymin = 1e9
    #ymax = 1e22
    #xmin = 1e-9
    #xmax = 1e-5
    # config for shh
    ymin = 1e-7
    ymax = 1e7
    xmin = 1e-9
    xmax = 1e-5
    # config for automatic
    #ymin = None
    #ymax = None
    #xmin = None
    #xmax = None
    a_of_x = False
    a_alterna = False
    detrend = True
    cut_averaged_head = 0
    which = "mean"
    shh_anal = True
    o_i = "o"
    anal_fit = True
    anal_fit_norm = False
    model_fit = False

    distance_to_river_list = [aquifer_length - i for i in obs_locations]
    ###############################################################################
    ###############################################################################


    """
    ###############################################################################
    # configurations for model runs:
    # /Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30_whitenoise
    # /Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30_mHM
    # /Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/frequency/dupuit_flow
    ###############################################################################

    path_to_multiple_projects = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30_mHM"
    project_folder_list = [
        f for f in os.listdir(str(path_to_multiple_projects)) if not f.startswith(".")
    ]
    try:
        project_folder_list.remove("fitting_results")
    except ValueError:
        pass
    project_folder_list.sort()
    obs_point_list = [
        "obs_0000",
        "obs_0010",
        "obs_0100",
        "obs_0200",
        "obs_0300",
        "obs_0400",
        "obs_0500",
        "obs_0600",
        "obs_0700",
        "obs_0800",
        "obs_0900",
        "obs_0950",
        "obs_0960",
        "obs_0970",
        "obs_0980",
        "obs_0990",
        "obs_1000",
    ]
    distance_to_river_list = [
        1000,
        990,
        900,
        800,
        700,
        600,
        500,
        400,
        300,
        200,
        100,
        50,
        40,
        30,
        20,
        10,
        0.1,
    ]

    Ss_list = [
        1.20e-03,
        1.10e-03,
        1.00e-03,
        9.00e-04,
        8.00e-04,
        7.00e-04,
        6.00e-04,
        5.00e-04,
        4.00e-04,
        3.00e-04,
        2.00e-04,
        1.00e-04,
        9.00e-05,
    ]
    S_list = [
        3.60e-02,
        3.30e-02,
        3.00e-02,
        2.70e-02,
        2.40e-02,
        2.10e-02,
        1.80e-02,
        1.50e-02,
        1.20e-02,
        9.00e-03,
        6.00e-03,
        3.00e-03,
        2.70e-03,
    ]
    kf_list = [
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
        1.00e-05,
    ]

    aquifer_thickness = 30
    aquifer_length = 1000
    weights_d = [1, 1, 1, 1, 1]
    a_d_in = None
    t_d_in = None
    # a_d_in=3e-8
    # t_d_in=6.8e+7
    time_steps = 8401
    time_step_size = 86400
    comment = "b1_"
    threshold = 1e-6
    fit = True
    mean_thick = False
    icsub = None
    target = False
    cutoff = None
    ymin = 1e9
    ymax = 1e22
    a_of_x=False
    a_alterna=False
    detrend=False
    ###############################################################################
    ###############################################################################
    """

    """
    ###############################################################################
    # configurations for model run: run1_20181030
    ###############################################################################

    path_to_multiple_projects = "/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30"
    project_folder_list = [f for f in os.listdir(str(path_to_multiple_projects)) if not f.startswith('.')]
    project_folder_list.sort()
    #obs_point_list = ["0000", "0200", "0400", "0500", "0600", "0800", "0950", "0990", "1000"]

    aquifer_thickness = [100, 100, 100, 100,
                         10, 10, 10, 10,
                         1, 1, 1, 1,
                         20, 20, 20, 20,
                         2, 2, 2, 2,
                         50, 50, 50, 50,
                         5, 5, 5, 5]

    #aquifer_length = 100
    #aquifer_length = 200
    #aquifer_length = 500
    #aquifer_length = 1000
    aquifer_length = 2000
    #aquifer_length = 5000

    #obs_point_list = ['obs_00010', 'obs_00040', 'obs_00060', 'obs_00080', 'obs_00090'] # 100
    #obs_point_list = ['obs_00020', 'obs_00080', 'obs_00120', 'obs_00160', 'obs_00180'] # 200
    #obs_point_list = ['obs_00050', 'obs_00200', 'obs_00300', 'obs_00400', 'obs_00450'] # 500
    #obs_point_list = ['obs_00100', 'obs_00400', 'obs_00600', 'obs_00800', 'obs_00900'] # 1000
    obs_point_list = ['obs_00200', 'obs_00800', 'obs_01200', 'obs_01600', 'obs_01800'] # 2000
    #obs_point_list = ['obs_00500', 'obs_02000', 'obs_03000', 'obs_04000', 'obs_04500'] # 5000

    #distance_to_river_list = [90, 60, 40, 20, 10] # 100
    #distance_to_river_list = [180, 120, 80, 40, 20] # 200
    #distance_to_river_list = [450, 300, 200, 100, 50] # 500
    #distance_to_river_list = [900, 600, 400, 200, 100] # 1000
    distance_to_river_list = [1800, 1200, 800, 400, 200]  # 2000
    #distance_to_river_list = [4500, 3000, 2000, 1000, 500] # 5000
    ###############################################################################
    ###############################################################################
    """

    ###############################################################################
    ###############################################################################

    # write parameters in log file

    print(
        "log-file for power spectral density of multiple OGS model runs"
        + "\n###############################################################################"
        + "\nParameters"
        + "\n----------"
        + "\nDirectory: "
        + str(path_to_multiple_projects)
        + "\nProject folder list: "
        + str(project_folder_list)
        + "\nObservation points: "
        + str(obs_point_list)
        + "\nDistances to river: "
        + str(distance_to_river_list)
        + "\nStorativities: "
        + str(S_list)
        + "\nspec. Storativities: "
        + str(Ss_list)
        + "\nHydraulic cond.: "
        + str(kf_list)
        + "\nAquifer thickness: "
        + str(aquifer_thickness)
        + "\nAquifer length: "
        + str(aquifer_length)
        + "\nWeights for fit: "
        + str(weights_d)
        + "\na_d: "
        + str(a_d_in)
        + "\nt_d: "
        + str(t_d_in)
        + "\nTime steps: "
        + str(time_steps)
        + "\nTime step size: "
        + str(time_step_size)
        + "\nComment: "
        + str(comment)
        + "\nMaximum x-value of PSD: "
        + str(threshold)
        + "\nFit: "
        + str(fit)
        + "\nSpat. averaged aquifer thickness fo rech time step: "
        + str(mean_thick)
        + "\nSubstraction of boundary conditions: "
        + str(icsub)
        + "\nPlot target model with OGS-input parameters: "
        + str(target)
        + "\nDelete the first # values of time series: "
        + str(cutoff)
        + "\nMinimum of y-achsis: "
        + str(ymin)
        + "\nMaximum of y-achsis: "
        + str(ymax)
        + "\nMinimum of x-achsis: "
        + str(xmin)
        + "\nMaximum of x-achsis: "
        + str(xmax)
        + "\nUse parameter 'a' for linear model in dependence of location: "
        + str(a_of_x)
        + "\nUse an alternative calculation for parameter 'a' for Dupuit model: "
        + str(a_alterna)
        + "\nDetrend the time series: "
        + str(detrend)
        + "\nCut the averaged head by the initial conditions: "
        + str(cut_averaged_head)
        + "\nWhere to grab the piezometric head along a vertical line as observation point (min, max, mean): "
        + str(which)
        + "\nPlot the analytical power spectrum for ogs input parameters: "
        + str(shh_anal)
        + "\nPlot input ('i'), output ('o') oder resulting ('oi') power spectra: "
        + str(o_i)
        + "\Use the analytical model to fit the PSD: "
        + str(anal_fit)
        + "\nDivide the spectrum by the input spectrum to optain normalized PSD for the fitted model: "
        + str(anal_fit_norm)
        + "\nUse the Dupuit and Linear reservoir model to fit the PSD: "
        + str(model_fit)
        + "\n###############################################################################"
    )


    ###############################################################################
    ###############################################################################


    # thresholds = [3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 1e-7, 2e-7, 8e-7]


    obs_index = 0  # index for iterating through the observation points
    i = 0  # index for different model runs
    j = 0  # index for different observation points
    k = 0  # index for different methods, PSD

    T_l = np.zeros((len(project_folder_list), len(obs_point_list), len(methods)))
    kf_l = np.zeros_like(T_l)
    Ss_l = np.zeros_like(T_l)
    D_l = np.zeros_like(T_l)
    a_l = np.zeros_like(T_l)
    t_l = np.zeros_like(T_l)
    T_d = np.zeros((len(project_folder_list), len(obs_point_list), len(methods)))
    kf_d = np.zeros_like(T_l)
    Ss_d = np.zeros_like(T_l)
    D_d = np.zeros_like(T_l)
    a_d = np.zeros_like(T_l)
    t_d = np.zeros_like(T_l)
    T_anal = np.zeros_like(T_l)
    S_anal = np.zeros_like(T_l)

    font = {"family": "normal", "weight": "normal", "size": 20}
    plt.rc("font", **font)
    plt.rc("legend",fontsize=15)

    # loop over all project directories
    for i, project_folder in enumerate(project_folder_list):
        print(
            "###############################################################################"
        )
        print("Starting with project: " + project_folder)
        path_to_project = path_to_multiple_projects + "/" + project_folder
        single_file_names = [
            f for f in os.listdir(str(path_to_project)) if f.endswith(".tec")
        ]
        single_file_names.sort()
        name_of_project_ogs = str(
            [f for f in os.listdir(str(path_to_project)) if f.endswith(".rfd")]
        )[2:-6]
        print(name_of_project_ogs)

        print("This is the name of the project: ", name_of_project_ogs)

        if mean_thick == True:
            # loop over all .tec files
            aquifer_thickness = np.mean(
                np.loadtxt(
                    path_to_multiple_projects
                    + "/"
                    + project_folder
                    + "/spatial_averaged_head_timeseries.txt"
                )[cut_averaged_head:]
            )
        for j, single_file_name in enumerate(single_file_names):
            obs_point = obs_point_list[j]
            print("1st: Getting time series for observation point: " + str(obs_point))
            fft_data, recharge = get_fft_data_from_simulation(
                path_to_project=path_to_project,
                name_of_project_ogs=name_of_project_ogs,
                single_file=path_to_project + "/" + single_file_name,
                time_steps=time_steps,
                obs_point=obs_point,
                which=which,
            )

            print(
                "LAST ENTRY OF HEAD and RECHARGE TIME SERIES WAS SET EQUAL TO PREVIOUS ONE DUE TO UNREASONABLE RESULTS"
            )
            fft_data[-1] = fft_data[-2]
            recharge[-1] = recharge[-2]

            if cutoff != None:
                print(
                    "First "
                    + str(cutoff)
                    + " data points in time series were deleted due to instationary of the aquifer."
                )
                fft_data = fft_data[cutoff:]
                recharge = recharge[cutoff:]

            if icsub != None:
                print(
                    "All values were substracted by "
                    + str(icsub)
                    + " which is the height of the stream (BC)."
                )
                fft_data = fft_data - icsub
                recharge = recharge - icsub

            for k, method in enumerate(methods):
                print("2nd: Calculating PSD...")
                print("Method: " + str(method))
                T_anal[i, j, k], S_anal[i, j, k], T_l[i, j, k], kf_l[i, j, k], Ss_l[i, j, k], D_l[i, j, k], a_l[i, j, k], t_l[
                    i, j, k
                ], T_d[i, j, k], kf_d[i, j, k], Ss_d[i, j, k], D_d[i, j, k], a_d[
                    i, j, k
                ], t_d[
                    i, j, k
                ], power_spectrum_output, power_spectrum_input, power_spectrum_output, power_spectrum_result, frequency_input = fft_psd(
                    fft_data=fft_data,
                    recharge=recharge,
                    # threshold=thresholds[j],
                    threshold=threshold,
                    path_to_project=path_to_project,
                    method=method,
                    fit=fit,
                    savefig=True,
                    saveoutput=True,
                    obs_point=obs_point,
                    single_file=path_to_project + "/" + single_file_name,
                    dupuit=True,
                    distance_to_river=distance_to_river_list[j],
                    #                                           aquifer_thickness=aquifer_thickness[i],
                    aquifer_thickness=aquifer_thickness,
                    aquifer_length=aquifer_length,
                    time_step_size=86400,
                    weights_d=weights_d,
                    comment=comment,
                    o_i=o_i,
                    a_d=a_d_in,
                    t_d=t_d_in,
                    Ss_list=Ss_list,
                    kf_list=kf_list,
                    obs_number=j,
                    model_number=i,
                    distance_to_river_list=distance_to_river_list,
                    target=target,
                    ymin=ymin,
                    ymax=ymax,
                    xmin=xmin,
                    xmax=xmax,
                    a_of_x=a_of_x,
                    a_alterna=a_alterna,
                    detrend=detrend,
                    shh_anal=shh_anal,
                    Sy=S_list[i],
                    T=kf_list[i] * aquifer_thickness,
                    anal_fit=anal_fit,
                    anal_fit_norm=anal_fit_norm,
                    model_fit=model_fit
                )

    # if i == 1:
    #   break

    params_anal = [T_anal, S_anal]
    names_anal = ["T", "S"]
    labels_anal = ["T [m2/s]", "S [-]"]

    params_l = [T_l, kf_l, Ss_l, D_l, a_l, t_l]
    params_d = [T_d, kf_d, Ss_d, D_d, a_d, t_d]
    names = ["T", "kf", "Ss", "D", "a", "tc"]
    labels = ["T [m2/s]", "kf [m/s]", "Ss [1/m]", "D [m2/s]", "a [-]", "tc [s]"]


    # create folder for plots if folder doesn't exist already
    path_to_results = path_to_multiple_projects + "/fitting_results/"
    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results)

    # save parameter
    np.save(path_to_results + str("_parameter_anal.npy"), params_anal)
    np.save(path_to_results + str("_parameter_linear.npy"), params_l)
    np.save(path_to_results + str("_parameter_dupuit.npy"), params_d)


    def plot(params_l=None, params_d=None, methods=methods, labels=labels):
        print("Created all PSDs. Continue with plotting of results from linear fitting...")

        params_anal = np.load(path_to_results + str("_parameter_anal.npy"))
        params_l = np.load(path_to_results + str("_parameter_linear.npy"))
        params_d = np.load(path_to_results + str("_parameter_dupuit.npy"))

        from matplotlib.lines import Line2D
        from own_colors import own_colors

        font = {"family": "normal", "weight": "normal", "size": 40}
        plt.rc("font", **font)
        plt.rc("legend",fontsize=30)


        if anal_fit == True:
            # analytical model
            l = 0  # index for different parameters
            m = 0  # index for different methods
            n = 0  # index for different model runs
            plt.ioff()
            for l, param in enumerate(params_anal):
                for m, method in enumerate(methods):
                    plt.figure(figsize=(30, 30))
                    for n, project_folder in enumerate(project_folder_list):
                        print(
                            "Analytical fit:"
                            + "\nParameter: "
                            + str(labels_anal[l])
                            + "\nobservation points: "
                            + str(obs_point_list)
                            + "\nDerived parameter value: "
                            + str(param[n, :, m])
                            + "\nmethod: "
                            + str(method)
                            + "\nproject folder: "
                            + str(project_folder)
                        )
                        # configuration for homo/vertical/horizontal
                        plt.semilogy(
                            obs_point_list,
                            param[n, :, m],
                            label=project_folder[-8:],
                            color=own_colors(n),
                            lw=5,
                        )
                        params_real = []
                        for o, obs_point in enumerate(obs_point_list):
                            params_real.append((T_list[n], S_list[n]))

                        plt.semilogy(
                            obs_point_list,
                            [params_real[x][l] for x in range(0, len(obs_point_list))],
                            ls="dashed",
                            lw=5,
                            color=own_colors(n),
                        )
                    plt.title(
                        "Method: "
                        + str(method)
                        + '\nFit: "analytical"'
                        + "\nParameter: "
                        + str(labels_anal[l])
                    )
                    Line2D(
                        [0],
                        [0],
                        color=own_colors(n),
                        linewidth=5,
                        linestyle="--",
                        label="model input value",
                    )
                    plt.grid(which="minor")
                    plt.grid(which="major")
                    if names_anal[l] == "T":
                        plt.ylim(min(T_list)/10, max(T_list)*10)
                    if names_anal[l] == "S":
                        plt.ylim(min(S_list)/10, max(S_list)*10)
                    #plt.ylim(param[np.isfinite(param[:, :, m])].min()/10, param[np.isfinite(param[:, :, m])].min()*10)
                    plt.xlabel("observation point")
                    plt.ylabel(str(labels_anal[l]))
                    plt.xticks(rotation=60)
                    plt.legend(loc="best", ncol=4)
                    print(
                        "Saving fig: ",
                        str(comment)
                        + str(names_anal[l])
                        + "_"
                        + str(threshold)
                        + "_"
                        + str(method)
                        + "_analytical"
                        + ".png",
                    )
                    plt.savefig(
                        str(path_to_results)
                        + str(comment)
                        + str(names_anal[l])
                        + "_"
                        + str(threshold)
                        + "_"
                        + str(method)
                        + "_analytical"
                        + ".png"
                    )
                    plt.close("all")

        if model_fit == True:
            # linear model
            l = 0  # index for different parameters
            m = 0  # index for different methods
            n = 0  # index for different model runs
            plt.ioff()
            for l, param in enumerate(params_l):
                for m, method in enumerate(methods):
                    plt.figure(figsize=(25, 25))
                    for n, project_folder in enumerate(project_folder_list):
                        print(
                            "Linear fit:"
                            + "\nParameter: "
                            + str(labels[l])
                            + "\nobservation points: "
                            + str(obs_point_list)
                            + "\nDerived parameter value: "
                            + str(param[n, :, m])
                            + "\nmethod: "
                            + str(method)
                            + "\nproject folder: "
                            + str(project_folder)
                        )
                        # plt.figure(figsize=(20, 16))
                        # configuration for homogeneous model runs
                        # plt.semilogy(obs_point_list, param[i,:,k], label=path_to_project[-12:-9])
                        # configuration for homo/vertical/horizontal
                        plt.semilogy(
                            obs_point_list,
                            param[n, :, m],
                            label=project_folder[-8:],
                            color=own_colors(n),
                            lw=3,
                        )
                        params_real = []
                        for o, obs_point in enumerate(obs_point_list):
                            if a_of_x == True:
                                params_real.append(
                                    calc_aq_param(
                                        Ss_list[n],
                                        kf_list[n],
                                        aquifer_length,
                                        aquifer_thickness,
                                        model="linear",
                                        distance=distance_to_river_list[o],
                                    )
                                )
                                print(
                                    "Calculating real model parameters for parameter 'a' in dependence on x"
                                    + "\nobservation point: "
                                    + str(obs_point)
                                    + "\nSs: "
                                    + str(Ss_list[n])
                                    + "\nkf: "
                                    + str(kf_list[n])
                                    + "\naquifer length: "
                                    + str(aquifer_length)
                                    + "\naquifer thickness: "
                                    + str(aquifer_thickness)
                                    + "\nmodel: "
                                    + "linear"
                                    + "\ndistance to river: "
                                    + str(distance_to_river_list[o])
                                    + "\nOutput: [T, kf, Ss, D, a, t] "
                                    + str(params_real[o])
                                )
                            else:
                                params_real.append(
                                    calc_aq_param(
                                        Ss_list[n],
                                        kf_list[n],
                                        aquifer_length,
                                        aquifer_thickness,
                                        model="linear",
                                    )
                                )
                                print(
                                    "Calculating real model parameters for parameter 'a' INdependent on x"
                                    + "\nobservation point: "
                                    + str(obs_point)
                                    + "\nSs: "
                                    + str(Ss_list[n])
                                    + "\nkf: "
                                    + str(kf_list[n])
                                    + "\naquifer length: "
                                    + str(aquifer_length)
                                    + "\naquifer thickness: "
                                    + str(aquifer_thickness)
                                    + "\nmodel: "
                                    + "linear"
                                    + "\nOutput: [T, kf, Ss, D, a, t] "
                                    + str(params_real[o])
                                )
                        plt.semilogy(
                            obs_point_list,
                            [params_real[x][l] for x in range(0, len(obs_point_list))],
                            ls="dashed",
                            lw=3,
                            color=own_colors(n),
                        )
                    plt.title(
                        "Method: "
                        + str(method)
                        + '\nFit: "linear"'
                        + "\nParameter: "
                        + str(labels[l])
                    )
                    Line2D(
                        [0],
                        [0],
                        color=own_colors(n),
                        linewidth=3,
                        linestyle="--",
                        label="model input value",
                    )
                    plt.grid(True)
                    plt.xlabel("observation point")
                    plt.ylabel(str(labels[l]))
                    plt.xticks(rotation=60)
                    plt.legend(loc="best", ncol=4)
                    print(
                        "Saving fig: ",
                        str(comment)
                        + str(names[l])
                        + "_"
                        + str(threshold)
                        + "_"
                        + str(method)
                        + "_linear"
                        + ".png",
                    )
                    plt.savefig(
                        str(path_to_results)
                        + str(comment)
                        + str(names[l])
                        + "_"
                        + str(threshold)
                        + "_"
                        + str(method)
                        + "_linear"
                        + ".png"
                    )
                    plt.close("all")

            # Dupuit model
            l = 0  # index for different parameters
            m = 0  # index for different methods
            n = 0  # index for different model runs
            plt.ioff()
            for l, param in enumerate(params_d):
                for m, method in enumerate(methods):
                    plt.figure(figsize=(25, 25))
                    for n, project_folder in enumerate(project_folder_list):
                        print(
                            "Dupuit fit:"
                            + "\nParameter: "
                            + str(labels[l])
                            + "\nobservation points: "
                            + str(obs_point_list)
                            + "\nDerived parameter value: "
                            + str(param[n, :, m])
                            + "\nmethod: "
                            + str(method)
                            + "\nproject folder: "
                            + str(project_folder)
                        )
                        # configuration for homogeneous model runs
                        # plt.semilogy(obs_point_list, param[i,:,k], label=path_to_project[-12:-9])
                        # configuration for homo/vertical/horizontal
                        plt.semilogy(
                            obs_point_list,
                            param[n, :, m],
                            label=project_folder[-8:],
                            color=own_colors(n),
                            lw=3,
                        )
                        params_real = []
                        for o, obs_point in enumerate(obs_point_list):
                            params_real.append(
                                calc_aq_param(
                                    Ss_list[n],
                                    kf_list[n],
                                    aquifer_length,
                                    aquifer_thickness,
                                    model="dupuit",
                                    distance=distance_to_river_list[o],
                                    a_alterna=a_alterna,
                                )
                            )
                            print(
                                "Calculating real model parameters for parameter 'a' in dependence on x"
                                + "\nobservation point: "
                                + str(obs_point)
                                + "\nSs: "
                                + str(Ss_list[n])
                                + "\nkf: "
                                + str(kf_list[n])
                                + "\naquifer length: "
                                + str(aquifer_length)
                                + "\naquifer thickness: "
                                + str(aquifer_thickness)
                                + "\nmodel: "
                                + "dupuit"
                                + "\ndistance to river: "
                                + str(distance_to_river_list[o])
                                + "\na_alterna: "
                                + str(a_alterna)
                                + "\nOutput: [T, kf, Ss, D, a, t] "
                                + str(params_real[o])
                            )
                        plt.semilogy(
                            obs_point_list,
                            [params_real[x][l] for x in range(0, len(obs_point_list))],
                            ls="dashed",
                            lw=3,
                            color=own_colors(n),
                        )
                    plt.title(
                        "Method: "
                        + str(method)
                        + '\nFit: "Dupuit"'
                        + "\nParameter: "
                        + str(labels[l])
                    )
                    plt.grid(True)
                    plt.xlabel("observation point")
                    plt.ylabel(str(labels[l]))
                    plt.xticks(rotation=60)
                    plt.legend(loc="best", ncol=4)
                    print(
                        "Saving fig: ",
                        str(comment)
                        + str(names[l])
                        + "_"
                        + str(threshold)
                        + "_"
                        + str(method)
                        + "_dupuit"
                        + ".png",
                    )
                    plt.savefig(
                        str(path_to_results)
                        + str(comment)
                        + str(names[l])
                        + "_"
                        + str(threshold)
                        + "_"
                        + str(method)
                        + "_dupuit"
                        + ".png"
                    )
                    plt.close("all")


    plot()



    if model_fit == True:
        # plotten der power models der power spectras for obtained parameters and real parameters
        # one plot for one model run and all observation points

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


        p = 0  # index for model runs
        q = 0  # index for observation point
        r = 0  # index for data points

        import scipy.fftpack as fftpack

        font = {"family": "normal", "weight": "normal", "size": 30}
        plt.rc("font", **font)
        plt.rc("legend", fontsize=15)

        time_step_size = 86400
        len_output = 8400
        frequency = (abs(fftpack.fftfreq(len_output, time_step_size))[: len_output / 2])[1:]

        params_d = np.load(path_to_results + str("_parameter_dupuit.npy"))

        T_d, kf_d, Ss_d, D_d, a_d, t_d = (
            params_d[0],
            params_d[1],
            params_d[2],
            params_d[3],
            params_d[4],
            params_d[5],
        )

        for p, project_folder in enumerate(project_folder_list):
            plt.figure(figsize=(20, 15))
            for q, obs in enumerate(obs_point_list):
                plt.axis([1e-9, 1e-5, 1e5, 1e19])
                params_real = calc_aq_param(
                    Ss_list[p],
                    kf_list[p],
                    aquifer_length,
                    aquifer_thickness,
                    distance=distance_to_river_list[q],
                    model="dupuit",
                    a_alterna=a_alterna,
                )
                plt.loglog(
                    frequency,
                    [
                        dupuit_fit(a_d=a_d[p, q, 0], t_d=t_d[p, q, 0], f_d=frequency[i])
                        for i in range(0, len(frequency))
                    ],
                    label=str(obs),
                    color=own_colors(q),
                )
                plt.loglog(
                    frequency,
                    [
                        dupuit_fit(a_d=params_real[4], t_d=params_real[5], f_d=frequency[i])
                        for i in range(0, len(frequency))
                    ],
                    ls="dashed",
                    color=own_colors(q),
                )
                plt.title(
                    "Dupuit Models - comparison of input and output parameter\nModel: "
                    + str(project_folder)[-8:]
                )
                plt.xlabel("frequency [1/s]")
                plt.ylabel("spectral power")
                plt.legend(loc="best", ncol=4)
            print(
                "saving model comparison image: "
                + "dupuit_"
                + str(comment)
                + str(project_folder)
                + ".png"
            )
            plt.savefig(
                str(path_to_results) + "dupuit_" + str(comment) + str(project_folder) + ".png"
            )
            plt.close("all")


        p = 0  # index for model runs
        q = 0  # index for observation point
        r = 0  # index for data points
        params_l = np.load(path_to_results + str("_parameter_linear.npy"))

        T_l, kf_l, Ss_l, D_l, a_l, t_l = (
            params_l[0],
            params_l[1],
            params_l[2],
            params_l[3],
            params_l[4],
            params_l[5],
        )

        for p, project_folder in enumerate(project_folder_list):
            plt.figure(figsize=(20, 15))
            for q, obs in enumerate(obs_point_list):
                plt.axis([1e-9, 1e-5, 1e5, 1e19])
                if a_of_x == True:
                    params_real = calc_aq_param(
                        Ss_list[p],
                        kf_list[p],
                        aquifer_length,
                        aquifer_thickness,
                        model="linear",
                        distance=distance_to_river_list[q],
                    )
                else:
                    params_real = calc_aq_param(
                        Ss_list[p],
                        kf_list[p],
                        aquifer_length,
                        aquifer_thickness,
                        model="linear",
                    )
                plt.loglog(
                    frequency,
                    [
                        linear_fit(a_l=a_l[p, q, 0], t_l=t_l[p, q, 0], f_l=frequency[i])
                        for i in range(0, len(frequency))
                    ],
                    label=str(obs) + "PSD fit output",
                )
                plt.loglog(
                    frequency,
                    [
                        linear_fit(a_l=params_real[4], t_l=params_real[5], f_l=frequency[i])
                        for i in range(0, len(frequency))
                    ],
                    ls="dashed",
                )
                plt.xlabel("frequency [1/s]")
                plt.ylabel("spectral power")
                plt.legend(loc="best", ncol=4)
                plt.title(
                    "Linear Reservoir Models - comparison of input and output parameter\nModel: "
                    + str(project_folder)[-8:]
                )
            print(
                "saving model comparison image: "
                + "linear_"
                + str(comment)
                + str(project_folder)
                + ".png"
            )
            plt.savefig(
                str(path_to_results) + "linear_" + str(comment) + str(project_folder) + ".png"
            )
            plt.close("all")


    now = time.time()
    elapsed = now - then
    print("Time elapsed: " + str(np.round(elapsed, 2)) + " s")




list_of_path_to_multiple_projects = [
    "/data/MultiMoSi/houben/spectral_analysis/20181210_darcy/1000/10",
    "/data/MultiMoSi/houben/spectral_analysis/20181210_darcy/1000/20",
    "/data/MultiMoSi/houben/spectral_analysis/20181210_darcy/1000/30",
    "/data/MultiMoSi/houben/spectral_analysis/20181210_darcy/2000/10",
    "/data/MultiMoSi/houben/spectral_analysis/20181210_darcy/2000/20",
    "/data/MultiMoSi/houben/spectral_analysis/20181210_darcy/2000/30",
    "/data/MultiMoSi/houben/spectral_analysis/20181210_darcy/5000/10",
    "/data/MultiMoSi/houben/spectral_analysis/20181210_darcy/5000/20",
    "/data/MultiMoSi/houben/spectral_analysis/20181210_darcy/5000/30"
]
list_of_aquifer_thicknesses = [10, 20, 30, 10, 20, 30, 10, 20, 30]
list_of_aquifer_lengths = [1000, 1000, 1000, 2000, 2000, 2000, 5000, 5000, 5000]
comment = "1_anal_"

for index, path in enumerate(list_of_path_to_multiple_projects):
    psd_multi_multi(path, list_of_aquifer_thicknesses[index], list_of_aquifer_lengths[index], comment)
