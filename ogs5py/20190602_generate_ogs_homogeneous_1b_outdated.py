## This script generates OGS setups for a heterogeneous, 2D, vertical model domain. Every model setup ist first generated with a steady state configuration, afterwards the OGS run starts, output is redirected into the "steady" folder and input files are changed to input files with transient settings.

import sys
import os
import numpy as np
from ogs5py import OGS, MPD, MSH
import shutil
from ogs5py.reader import readpvd, readtec_point
from gstools import SRF, Gaussian
import matplotlib.pyplot as plt
import platform

# ------------------------ogs5py configurations---------------------------------- #
# get the current working directory
# CWD = os.getcwd()
sys.argv[1] = "/Users/houben/phd/python/scripts/ogs5py"
CWD = sys.argv[1]
# the name of this script
file_name = sys.argv[0]
# ------------------------domain configuration-------------------------------- #
length = 1000
thickness = 30
n_cellsx = int(length)
n_cellsz = int(thickness)
# ------------------------time configuration---------------------------------- #
time_start = 0
time_steps = np.array([500])
step_size = np.array([86400])
time_end = np.sum(time_steps*step_size)
# ------------------------ogs configuration---------------------------------- #
# name of the folder of one single ogs run
storage = 0.01
recharge_list = [1.1574074074074073e-08, 1.1574074074074073e-07]
# transmissivity values for each block
kf = 0.001

name='1b_steady_gw_flow_' + 'stor_' + str(storage) + "_kf_" + str(kf)
dim_no = 2
parent_dir = CWD + '/setup'
# name of directory (entire path) of one single ogs run
dire = parent_dir + '/' + name
# make folders
if not os.path.exists(parent_dir):
    os.mkdir(parent_dir)
if not os.path.exists(dire):
    os.mkdir(dire)
pcs_type_flow = 'GROUNDWATER_FLOW'
var_name_flow = 'HEAD'
t_id = 'transect'
# ------------------------generate ogs base class----------------------------- #
ogs = OGS(task_root=dire+"/",
      task_id=t_id,
      output_dir=dire+"/"+"steady"+"/")
# ------------------------  MSH -------------------------------- #
# generate a rectangular mesh in x-z-plane
# first part of mesh, configuration based on an element size of 1x1 m, because variable "length" is used for number of nodes
ogs.msh.generate("rectangular", dim=2,
             mesh_origin=(0., 0.),
             element_no=(n_cellsx, n_cellsz),
             element_size=(1, 1))
ogs.msh.MATERIAL_ID = 0
# rotate mesh to obtain a cross section in x-z-plane
ogs.msh.rotate(angle=np.pi/2.0, rotation_axis=(1., 0., 0.))
# round nodes
ogs.msh.NODES[:, 1] = np.around(ogs.msh.NODES[:, 1], 0)
ogs.msh.NODES[:, 0] = np.around(ogs.msh.NODES[:, 0], 4)
ogs.msh.NODES[:, 2] = np.around(ogs.msh.NODES[:, 2], 4)
#            ogs.msh.export_mesh("/Users/houben/phd/modelling/20190531_SA_hetero_block/" + str(border) + "_msh.msh")
# ------------------------  GLI -------------------------------- #
ogs.gli.add_points(points=[[0., 0., 0.],
                       [length, 0., 0.],
                       [length, 0., thickness],
                       [0., 0., thickness]],
               names=['A', 'B', 'C', 'D'])
# generate polylines from points as boundaries: always define polylines along positive x,y,z
ogs.gli.add_polyline(name='bottom', points=['A', 'B'])
ogs.gli.add_polyline(name='right', points=['B', 'C'])
ogs.gli.add_polyline(name='top', points=['D', 'C'])
ogs.gli.add_polyline(name='left', points=['A', 'D'])
# add the points and polylines based on the aquifer length
obs = []
percents_of_length = [0,0.1,0.5,0.9,1]
for percent in percents_of_length:
    obs_loc = int(np.around(length*percent,0))
    obs_str = 'obs_' + str(obs_loc).zfill(5)
    obs.append(obs_str)
    obs.sort()
    ogs.gli.add_points(points=[obs_loc, 0., 0.], names=str('obs_' + str(obs_str) + '_bottom'))
    ogs.gli.add_points(points=[obs_loc, 0., thickness], names=str('obs_' + str(obs_str) + '_top'))
    ogs.gli.add_polyline(name=obs_str, points=[[obs_loc, 0., 0.], [obs_loc, 0., thickness]])

# --------------generate .rfd ------------------------- #
# 50 days of 1 mm/d, 150 days of 10 mm/d
time = np.arange(0,time_steps*(step_size+1), step_size)
recharge = []
for i in time:
    if i < (step_size * 51):
        recharge.append(recharge_list[0])
    else:
        recharge.append(recharge_list[1])
rfd_data = np.column_stack((time,recharge))
# write array to .rfd, wenn mainkeyword als keyword in der funktion genutzt wird.
ogs.rfd.add_block(CURVES=rfd_data)
# --------------generate different ogs input classes------------------------- #

# --------------    BC  ------------------------- #
ogs.bc.add_block(PCS_TYPE=pcs_type_flow,
             PRIMARY_VARIABLE=var_name_flow,
             GEO_TYPE=[['POLYLINE', 'right']],
             DIS_TYPE=[['CONSTANT', thickness]])
# --------------    IC  ------------------------- #
ogs.ic.add_block(PCS_TYPE=pcs_type_flow,
             PRIMARY_VARIABLE=var_name_flow,
             GEO_TYPE='DOMAIN',
             DIS_TYPE=[['CONSTANT', thickness]])
# --------------    MFP ------------------------- #
ogs.mfp.add_block(FLUID_TYPE='WATER',
              DENSITY=[[1, 0.9997e+3]],
              VISCOSITY=[[1, 1.309e-3]])
# --------------    MMP ------------------------- #
ogs.mmp.add_block(GEOMETRY_DIMENSION=dim_no,
              STORAGE=[[1, storage]],
              PERMEABILITY_TENSOR=[['ISOTROPIC', kf]],
              #PERMEABILITY_DISTRIBUTION=ogs.task_id+'.mpd',
              #POROSITY='0.35'
              )
# --------------    NUM ------------------------- #
ogs.num.add_block(PCS_TYPE=pcs_type_flow,
              # method error_tolerance max_iterations theta precond storage
              LINEAR_SOLVER=[[2, 1, 1.0e-10, 1000, 1.0, 100, 4]],
              ELE_GAUSS_POINTS=3,
              #NON_LINEAR_ITERATION=[['PICARD', 'ERNORM', 20, 0, 1e-6]]
              )
# --------------    OUT ------------------------- #
ogs.out.add_block(PCS_TYPE=pcs_type_flow,
              NOD_VALUES=[[var_name_flow],
                          ['VELOCITY_X1'],
                          ['VELOCITY_Z1']],
              ELE_VALUES=[['VELOCITY1_X'],
                          ['VELOCITY1_Z']],
              GEO_TYPE='DOMAIN',
              DAT_TYPE='PVD',
              TIM_TYPE=[['STEPS', 100]])

# set the output for every observation point
for obs_point in obs:
    ogs.out.add_block(PCS_TYPE=pcs_type_flow,
                      NOD_VALUES=[[var_name_flow],
                                  ['VELOCITY_X1'],
                                  ['VELOCITY_Z1']],
                      GEO_TYPE=[['POLYLINE', obs_point]],
                      DAT_TYPE='TECPLOT',
                      TIM_TYPE=[['STEPS', 1]])
# run for steady first, then transient
for state in ['steady','transient']:
    # --------------    ST  ------------------------- #
    if state == 'transient':
        ogs.st.reset()
        ogs.st.add_block(PCS_TYPE=pcs_type_flow,
                         PRIMARY_VARIABLE=var_name_flow,
                         GEO_TYPE=[['POLYLINE', 'top']],
                         DIS_TYPE=[['CONSTANT_NEUMANN', 1]],
                         TIM_TYPE=[['CURVE',1]])
    if state == 'steady':
        ogs.st.add_block(PCS_TYPE=pcs_type_flow,
                        PRIMARY_VARIABLE=var_name_flow,
                        GEO_TYPE=[['POLYLINE', 'top']],
                        DIS_TYPE=[['CONSTANT_NEUMANN', recharge_list[0]]])

    # --------------    PCS ------------------------- #
    if state == 'transient':
        ogs.pcs.reset()
        ogs.pcs.add_block(PCS_TYPE=pcs_type_flow,
                          NUM_TYPE='NEW',
                          PRIMARY_VARIABLE=var_name_flow,
                          RELOAD=[[2,1]],
                          BOUNDARY_CONDITION_OUTPUT=[[]])
    if state == 'steady':
        ogs.pcs.add_block(PCS_TYPE=pcs_type_flow,
                          TIM_TYPE='STEADY',
                          NUM_TYPE='NEW',
                          PRIMARY_VARIABLE=var_name_flow,
                          RELOAD=[[1,1]],
                          BOUNDARY_CONDITION_OUTPUT=[[]])
    # --------------    TIM ------------------------- #
    if state == 'transient':
        ogs.tim.reset()
        ogs.tim.add_block(PCS_TYPE=pcs_type_flow,
                          TIME_START=time_start,
                          TIME_END=time_end,
                          TIME_STEPS=zip(time_steps, step_size))

    # --------------run OGS simulation------------------------------------------- #
    ogs.write_input()
    if state == 'steady':
        file = open(dire+"/"+t_id+'.tim', 'w')
        file.write('#STOP')
        file.close()
    if state == 'steady':
        print("calculating steady state...")
        if platform.system() == 'Darwin':
            # path for local Mac
            ogs.run_model(ogs_root='/Users/houben/phd/ogs5/sources_bugfix_RWPT/ogs5/build/bin/ogs')
        if platform.system() == 'Linux':
            # path for eve
            ogs.run_model(ogs_root='/home/houben/OGS_source/ogs')


#print("run model")
#ogs.run_model(ogs_root='/Users/houben/PhD/ogs5/executable/ogs5_bugfix_RWPT')

#from ogs5py.reader import readtec_polyline
#tecs = readtec_polyline(task_id=name_of_project_ogs,task_root=path_to_project)

# shutil.copyfile(str(CWD) + file_name, parent_dir + file_name)

print('OGS-model generation finished!')
