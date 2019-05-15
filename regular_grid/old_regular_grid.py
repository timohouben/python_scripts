#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:12:42 2017

@author: sebastian
"""

from __future__ import print_function, division

import ogs5py_mesh as ogs

# from ogs5py_mesh import generator as gen
import numpy as np

mesh = ogs.mesh()
mesh.load("transect_01_script2_fine_ogs.msh")

print(mesh.NODES)

knopjes = mesh.NODES[:]
np.savetxt("knopjes.txt", knopjes)


source_begin = 13  # Anzahl der händischen Punkte
source_end = 511  #
length = source_end - source_begin
inter_begin = 611
inter_end = inter_begin + length
rest_begin = 1209
rest_end = rest_begin + length
max_nodes = 50609 + 1

rotations = int((max_nodes - rest_begin) / length)
print(rotations)


mesh.NODES[inter_begin:inter_end, 0] = mesh.NODES[source_begin:source_end, 0]

i = 0
while i < rotations:
    start = rest_begin + i * (length + 1)
    end = rest_end + i * (length + 1)
    mesh.NODES[start : end + 1, 0] = mesh.NODES[source_begin : source_end + 1, 0]
    i = i + 1


# mesh.NODES[:, 0] = np.around(mesh.NODES[:, 0], 1)
# mesh.NODES[:, 2] = np.around(mesh.NODES[:, 2], 6)

mesh.show()
# mesh.save("RESULT.msh")
