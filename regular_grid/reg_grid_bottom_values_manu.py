#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:30:15 2018

@author: houben
"""

from __future__ import print_function, division
import time
then = time.time()
import ogs5py_mesh as ogs
import glob
import numpy as np
#from ogs5py_mesh import generator as gen

#timeit.timeit('text.find(char)', setup='text = "sample string"; char = "g"')

#### grab all files ending with .msh
msh_files = glob.glob('*.msh')
#### sort the list with file names
msh_files.sort()

### Variable: number of nodes in x-direction, i.e. number of resulting classes: 
xnodes = 4001
### length of model in m (x-direction)
length = 1000




for name_mesh in msh_files:
    filename = name_mesh
    mesh = ogs.mesh()
    mesh.load(filename)
    #print(mesh.NODES)
    
    #print(mesh.NODES[1:5])
    
    #### define variable
    bottom = []             # nodes of bottom of mesh (y, z = 0)
    distance = []           # values of distance between nodes
    threshold_down = []     # lower threshold of classification
    threshold_up = []       # upper threshold of classification
    
    
    #### generate list with points from bottom (z = 0)
    #for i, line in enumerate(mesh.NODES):
    #    if line[2] == 0: 
    #        bottom.append(line[0])


    bottom = list(np.linspace(0, length, xnodes))
    bottom.sort()
    
    threshold_down[:] = [x - ((length/(xnodes-1)/2)-0.000000001) for x in bottom]
    threshold_up[:] = [x + ((length/(xnodes-1)/2)) for x in bottom]
    
    '''
    #### generate list with distances
    for i, line in enumerate(bottom):
        if i>0:
            distance.append(bottom[i] - bottom[i-1])
            
    #### add last value of list(distance) again to have enough classes
    distance.append(distance[len(distance)-1])
    
    #### define threshold by half of distance (classes have overlay on boarders)
    for i, line in enumerate(bottom, 0):
        threshold_down.append(line - distance[i]/2)
        threshold_up.append(line + distance[i]/2)
    '''
        
    print("Mesh correction in progress...")
    
    #### for each node in .msh define which class and substitute value with value of bottom respectively
    progress = np.size(mesh.NODES,0)
    #progress = range(mesh.NODES)
    for i, linei in enumerate(mesh.NODES):
        for j, linej in enumerate(bottom):
            if linei[0] > threshold_down[j] and linei[0] < threshold_up[j]:
                mesh.NODES[i,0] = bottom[j]
        if i == progress // 4:
            now = time.time()
            print('25% finished... run time: ' + str(now - then) + ' seconds')
        elif i == progress // 2:
            now = time.time()
            print('50% finished... run time: ' + str(now - then) + ' seconds')
        elif i == progress // 4 * 3:
            now = time.time()
            print('75% finished... run time: ' + str(now - then) + ' seconds')
        
               
            
            
            
        #print(i)            
                
    mesh.save(filename[:-4] + "_corr_bottom_values_manu.msh")
    now = time.time()
    print("Your script ran " + str(now - then) + " seconds.")
