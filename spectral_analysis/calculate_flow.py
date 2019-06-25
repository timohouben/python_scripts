def get_baseflow_from_polyline(
    task_id, task_root, single_file, orientation="vertical", save_flow=True
):
    """
    Calculate the baseflow based on a vertical polyline from an OGS run.
    The file type should be .tec and it must contain VELOCITY_X1 values.

    Parameters
    ----------

    task_id : string
        Name of the ogs task.
    task_root : string
        Directory of ogs model run.
    single_file : string
        Path to .tec file from the polyline.
    orientation : string (vertical, horizontal)
        NO FUNCTION YET
    save_flow : bool
        True: A txt-file will be saved in the root directoy.

    Yields
    ------
    flow_timeseries : 1D array
        A value for the outflow over the whole polyline for each time step.
    """

    from ogs5py.reader import readtec_polyline
    import numpy as np
    import os

    # read only 1 tec file for polyline for which the flow should be calculated
    tec = readtec_polyline(
        task_id=task_id, task_root=task_root, single_file=single_file
    )
    # time_steps = number of timesteps + initial values
    try:
        time_steps = tec["VELOCITY_X1"].shape[0]
        # nodes = number of nodes along polyline
        nodes = tec["VELOCITY_X1"].shape[1]
        flow_array = np.zeros((nodes, time_steps))
        flow_timeseries = []

        for i in range(0, time_steps):
            print("Time step " + str(i) + "of " + str(time_steps) + "...")
            # get the node values of the velocities for ith time step
            node_velocity = tec["VELOCITY_X1"][i, :]
            # get the node values of the distances measured from starting point of polyline
            # (i.e. y-value) for ith time step
            node_dist = tec["DIST"][i, :]
            flow_per_timestep = []
            # add flow for first node, calculate the distance between
            # 1st and 2nd node and multiply with first node value
            flow_per_timestep.append(
                (node_dist[1] - node_dist[0]) / 2 * node_velocity[0]
            )
            for j in range(0, len(node_velocity) - 2):
                # 1st loop: calculate the distance between first and second node
                diff_low = node_dist[j + 1] - node_dist[j]
                # 1st loop: calculate the distance between second and third node
                diff_up = node_dist[j + 2] - node_dist[j + 1]
                # 1st loop: multiply second node velocity with half of distance between
                # 1st and 2nd node and half of distance between 2nd and 3rd node
                flow_per_timestep.append(
                    (diff_low + diff_up) / 2 * node_velocity[j + 1]
                )
            # add flow for last node, multiply node velocity with half of distance between 2nd last and last node
            flow_per_timestep.append(
                ((node_dist[-1] - node_dist[-2]) / 2 * node_velocity[-1])
            )
            flow_per_timestep = np.asarray(flow_per_timestep)
            flow_array[:, i] = flow_per_timestep
        flow_timeseries = flow_array.sum(axis=0)
        # return flow_timeseries
    except KeyError:
        print(
            "ERROR: There is no VELOCITY_X1 time series in the given tec-file."
        )
    if save_flow == True:
        np.savetxt(single_file[:-4] + "_flow_timeseries.txt", flow_timeseries)
    return flow_timeseries


def plot_recharge_vs_baseflow(task_root, flow_timeseries, aquifer_length=None, calculate_tc_tr=False, percent=0.95, normalizer=86400):
    """
    Plot the discharge from the model with the recharge extracted from the rfd file.

    Parameters
    ----------
    task_root : string
        Directory of ogs model run.
    flow_timeseries : 1D array
        Output of get_baseflow_from_polyline()
    aquifer_length : float, default=None
        If a value is provided the script calculates the sum of inflow and
        outflow in the modelling period and prints it to the stdout.
    calculate_tc_tr : bool, default=False
        If True, tc and tr are calculated based on the first time step when
        recharge has reached the maximum.
    percet : float between 0 and 1, default: 0.95
        The percentage of discharge from recharge which should be considered
        for tc.
    normalizer : float, dafault=86400
        Time will be devided by this value.

    """

    import matplotlib.pyplot as plt
    from transect_plot import extract_rfd
    import numpy as np
    import os

    rfd = extract_rfd(path=task_root, export=False)

    if aquifer_length != None:
        q_total = np.sum(flow_timeseries)
        print("Total discharge rate in modelling period: {:.5f} [m^2/s]".format(q_total))
        r_total = np.sum(rfd[1][1:] * aquifer_length)
        print("Total recharge rate in modelling period: {:.5f} [m^2/s]".format(r_total))
        print(
            "{:.5f} % of recharge has reached the outflow of the model.".format(
                q_total / r_total
            )
        )

        if calculate_tc_tr == True:
            for i, item in enumerate(rfd[1]):
                if item == np.max(rfd[1]):
                    start_time = rfd[0][i] / normalizer
                    print("Time of max recharge: " + str(start_time))
                    break

            #switch(recharge_case):
            #case 1:
            # calculation of tc based on 95 % of recharge volume has reached the outflow
            sum_recharge = 0
            sum_discharge = 0
            for i, (item, jtem) in enumerate(zip(rfd[1]*aquifer_length, flow_timeseries)):
                if rfd[0][i]/normalizer > start_time and jtem >= percent*item:
                    stop_time = rfd[0][i] / normalizer
                    delta_t = stop_time - start_time
                    print("Discharge has reached " + str(percent) + " % of recharge at time: " + str(stop_time) + ". Delta t = " + str(delta_t))
                    break
                if i == len(flow_timeseries)-1:
                    print("Discharge has not reached " +str(percent) + " % of recharge.")

    elif aquifer_length == None and calculate_tc_tr == True:
        print("You need to specify the aquifer length to calculate tc and tr.")

    time = rfd[0][:] / normalizer
    fig, ax1 = plt.subplots()
    ax1.set_title(os.path.basename(task_root))
    ax1.plot(time[:len(flow_timeseries)], flow_timeseries, "b", linewidth=4)
    ax1.set_xlabel("time [d]")
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel("discharge $[m^2/s]$", color="b")
    ax1.tick_params("y", colors="b")
    y_lims = (0,0.0002)
    ax1.set_ylim((0, 0.0002))
    ax2 = ax1.twinx()

    if aquifer_length == None:
        ax2.plot(time, rfd[1][:], "r")
        ax2.set_ylabel("recharge $[m/s]$", color="r")
    elif aquifer_length != None:
        ax2.plot(time,rfd[1][:]*aquifer_length, "r", linewidth=2)
        ax2.set_ylabel("recharge $[m^2/s]$", color="r")
        ax2.set_ylim((0, y_lims[1]))
        if 'stop_time' in locals():
            plt.vlines(x=[start_time,stop_time], ymin=0, ymax=y_lims[1], color='g')
            plt.annotate("$\Delta t$ = " + str(delta_t), (stop_time + 1 ,0.8*y_lims[1]))
    ax2.tick_params("y", colors="r")
    fig.tight_layout()
    plt.savefig(task_root + "/baseflow_vs_recharge.png", dpi=300)
    plt.show()



if __name__ == "__main__":

    task_id = "transect"
    task_root = "/Users/houben/phd/modelling/20190602_Tc_vs_Tr/transport/setup/2a_transportstor_0.01_kf_0.001"
    single_file = task_root + "/" + "transect_ply_obs_01000_t5.tec"
    orientation = "vertical"
    aquifer_length = 1000

    flow_timeseries = get_baseflow_from_polyline(
        task_id=task_id,
        task_root=task_root,
        single_file=single_file,
        orientation="vertical",
        )

    plot_recharge_vs_baseflow(task_root=task_root, flow_timeseries=flow_timeseries, aquifer_length=1000, calculate_tc_tr=True, percent=0.95)

    ## to execute for multiple folders in a directory: dir has to contain the complete path of every folder
    #for dir in roots:
    #    flow_timeseries = get_baseflow_from_polyline(
    #    task_id="transect",
    #    task_root=dir,
    #    single_file=dir + "/transect_ply_obs_01000_t5_GROUNDWATER_FLOW.tec",
    #    orientation="vertical",)
    #    plot_recharge_vs_baseflow(task_root=dir, flow_timeseries=flow_timeseries, aquifer_length=1000, calculate_tc_tr=True, percent=0.95)
