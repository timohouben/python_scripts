def extract_timeseries(path, which="mean", process="GROUNDWATER_FLOW", rfd=1):
    """
    Function to extract time series for each observation point and store them as .txt.

    Parameters
    ----------

    path : string
        path to ogs directory
    which : string, 'mean', 'max', 'min'
        Which value should be taken from the vertical polyline of obs point.
    rfd : int
        Extract the #rfd curve from the rfd file and save it as txt. rfd = 0 : no extraction

    Yields
    ------

    .txt from every observation point with head values

    """

    import numpy as np
    from ogs5py.reader import readtec_polyline
    import glob

    # glob the name of the ogs run
    string = str(glob.glob(path + '/*.bc'))
    pos1 = string.rfind("/")
    task_id = string[pos1+1:-5]

    # extract the series given in the rfd file
    if rfd != 0:
        from ogs5py import OGS
        ogs = OGS(task_root = path + "/", task_id = task_id)
        ogs.rfd.read_file(path = path + "/" + task_id + ".rfd")
        #print(ogs.rfd.get_block(rfd-1)[''])
        values = np.asarray([values[1] for values in ogs.rfd.get_block(rfd-1)['']])
        time = np.asarray([time[0] for time in ogs.rfd.get_block(rfd-1)['']])
        np.savetxt(str(path) + '/' + 'rfd_curve#'+ str(rfd) + '.txt', values)
        np.savetxt(str(path) + '/' + 'time' + '.txt', time)

    # read all tec files
    print("Reading tec-files from " + path)
    tecs = readtec_polyline(task_id=task_id,task_root=path)

    # extract the time series and save them as .txt
    for obs in tecs['GROUNDWATER_FLOW'].keys():
        time_steps = len(tecs['GROUNDWATER_FLOW'][obs]['TIME'])
        number_of_columns = tecs[process][obs]["HEAD"].shape[1]
        if which == 'max':
            # select the maximum value (i.e. the uppermost) of polyline as long as polylines are defined from bottom to top
            head_ogs_timeseries_each_obs = tecs[process][obs]["HEAD"][:,number_of_columns-1]
        elif which == 'min':
            # select the minimum value (i.e. the lowermost) of polyline as long as polylines are defined from bottom to top
            head_ogs_timeseries_each_obs = tecs[process][obs]["HEAD"][:,0]
        elif which == 'mean':
            head_ogs_timeseries_each_obs=[]
            for step in range(time_steps):
                # calculates the mean of each time step
                head_ogs_timeseries_each_obs.append(np.mean(tecs[process][obs]["HEAD"][step,:]))
            head_ogs_timeseries_each_obs = np.asarray(head_ogs_timeseries_each_obs)
        np.savetxt(str(path) + '/' + 'head_ogs_' + str(obs) + '_' + str(which) + '.txt', head_ogs_timeseries_each_obs)
    return time, values

def plot_head_timeseries_vs_recharge(path, which="mean"):
    """
    Plot the head.txt with recharge time series on 2nd axis.
    Assuming that recharge is given in m/s and the time steps are in seconds with a dayli increment.
    """

    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import os.path

    # glob the name of the .txt files
    file_names_obs_list = glob.glob(path + '/*obs*' + which + '*.txt')
    file_names_obs_list.sort()
    file_name_rfd = str(glob.glob(path + '/*rfd*.txt')[0])
    file_name_time = str(glob.glob(path + '/*time*.txt')[0])

    rfd = np.loadtxt(file_name_rfd)
    time = np.loadtxt(file_name_time)
    # calculate mm/day from m/s
    rfd = rfd * 86400 * 1000
    time = time / 86400

    # first axis for recharge
    fig, ax1 = plt.subplots(figsize=(14, 10))
    plt.title("head timeseries at different observations points")
    color = 'tab:blue'
    ax1.set_xlabel('time [day]')
    ax1.set_ylabel('recharge [mm/day]', color=color)  # we already handled the x-label with ax1
    ax1.bar(time, rfd, width=1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0,2.5)
    #ax1.set_ylim([min(recharge), max(recharge)*2])
    #ax1.set_yticks([0, 1, 2, 3, 4, 5])

    # second axis for head
    ax2 = ax1.twinx()
    #ax2.set_ylim(30,30.1)
    #ax2.set_yticks(np.arange(26,40,0.5))
    color = 'tab:red'

    print("plotting...")
    if which == 'min':
        for obs in file_names_obs_list:
            # derive the head for the given observation point from ogs
            head_ogs = np.loadtxt(obs)
            ax2.plot(time, head_ogs, label = str(obs)[-13:-5] + ' OGS',  linestyle='-')
    elif which == 'max':
        for obs in file_names_obs_list:
            # derive the head for the given observation point from ogs
            head_ogs = np.loadtxt(obs)
            ax2.plot(time, head_ogs, label = str(obs)[-13:-5] + ' OGS',  linestyle='-')
    elif which == 'mean':
        for obs in file_names_obs_list:
            # derive the head for the given observation point from ogs
            head_ogs = np.loadtxt(obs)
            ax2.plot(time, head_ogs, label = str(obs)[-14:-4] + ' OGS',  linestyle='-')

        ax2.set_ylabel('head [m]', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(color='grey', linestyle='--', linewidth=0.5, which='both')
        handles, labels = ax2.get_legend_handles_labels()
        ax1.legend(handles, labels, loc=6, facecolor="white", framealpha=100)

        fig.tight_layout()

        # make a string from list obs_per_plot
        fig.savefig(str(path) + '/' + str(os.path.basename(str(path))) + '_' + str(which) + '_' + ".png")
        plt.close('all')
