# Specify the number ob slots manually! In case you run the mpi_local_generate_ogs.py the number of cores has to be the same in both scripts
#!/bin/bash

#$ -S /bin/bash
#$ -N gen_190531
#$ -o /work/houben/20190531_SP_hetero_block
#$ -j y
#$ -l h_rt=3600
#$ -l h_vmem=8G

#$ -pe openmpi-orte 4

# load modules for open mpi
module use /software/easybuild-E5-2690v4/modules/all/Core
module load uge/8.5.5-easybuild
module load foss/2018b

# load modules for python
module load python/3.6.2
module load libGLU

# load modules for mpi4py
module load openmpi/gcc/2.1.1-1
module load gcc/4/9.4-3

# activate python environment
source /home/houben/pyenv3.6.2/bin/activate

# mpirun with script to execute
mpirun -np 100 python3 /home/houben/python/scripts/ogs5py/20190531_generate_ogs_hetero_block_EVE.py /work/houben/20190531_SP_hetero_block 100

# run this stuff to surpress fork() warning
# mpirun --mca mpi_warn_on_fork 0 -np 100 python3 /home/houben/python/scripts/ogs5py/TEST/20190531_generate_ogs_hetero_block_EVE_Sebastian.py /home/houben/python/scripts/ogs5py/TEST/ 4
