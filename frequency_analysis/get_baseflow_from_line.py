def get_baseflow_from_line(path_to_file="/Users/houben/PhD/modelling/transect/ogs/confined/transient/rectangular/Groundwater@UFZ/Model_Setup_D_day_EVE/homogeneous/D18-D30_whitenoise/Groundwater@UFZ_eve_HOMO_276_D_25_results/transect_01_ply_obs_1000_t17_GROUNDWATER_FLOW.tec"):
    '''
    - Polyline has to be vertical! Only x-component of node-velocity is used.
    - All points in the line have to have the exact same x-coordinate (e.g. on a vertical domain border)

    '''

    from ogs5py.reader import readtec_polyline
    import numpy as np
    import os



    # read only 1 tec file for polyline for which the flow should be calculated
    tecs = readtec_polyline(single_file=path_to_file)

    # time_steps = number of timesteps + initial values
    try:
        time_steps = tecs["VELOCITY_X1"].shape[0]-1
        # nodes = number of nodes along polyline
        nodes = tecs["VELOCITY_X1"].shape[1]
        flow_array = np.zeros((nodes, time_steps))
        flow_timeseries = []

        for i in range(0,time_steps):
            # get the node values of the velocities for ith time step
            node_velocity = tecs["VELOCITY_X1"][i,:]
            # get the node values of the distances measured from starting point of polyline
            # (i.e. y-value) for ith time step
            node_dist = tecs["DIST"][i,:]
            flow_per_timestep = []
            # add flow for first node, calculate the distance between
            # 1st and 2nd node and multiply with first node value
            flow_per_timestep.append((node_dist[1] - node_dist[0]) / 2 * node_velocity[0])
            for j in range(0,len(node_velocity)-2):
                # 1st loop: calculate the distance between first and second node
                diff_low = node_dist[j+1] - node_dist[j]
                # 1st loop: calculate the distance between second and third node
                diff_up = node_dist[j+2] - node_dist[j+1]
                # 1st loop: multiply second node velocity with half of distance between
                # 1st and 2nd node and half of distance between 2nd and 3rd node
                flow_per_timestep.append((diff_low + diff_up) / 2 * node_velocity[j+1])
            # add flow for last node, multiply node velocity with half of distance between 2nd last and last node
            flow_per_timestep.append(((node_dist[-1] - node_dist[-2]) / 2 * node_velocity[-1]))
            flow_per_timestep = np.asarray(flow_per_timestep)
            flow_array[:,i] = flow_per_timestep
        flow_timeseries = flow_array.sum(axis=0)
        np.savetxt(str(path_to_file)[:-len(os.path.basename(str(path_to_file)))] + "/baseflow.txt", flow_timeseries)
        return flow_timeseries
    except KeyError:
        print('ERROR: There is no VELOCITY_X1 time series in the given tec-file.')

baseflow =get_baseflow_from_line()
import matplotlib.pyplot as plt
plt.plot(baseflow)
