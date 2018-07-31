#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:14:41 2018

@author: houben

script to run multiple ogs runs in a directory

Script will search for all .gli files in directory and subdirectory in CURRENT WORKING DIRECTIORY!
"""

import os
import os.path
import datetime

cwd = os.getcwd()

for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(".gli")]:
        #print os.path.join(dirpath, filename)
        time = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
        print("Start time: "+str(time))
        print("OGS Run : "+str(dirpath[1:])+"/"+str(filename[:-4]))
        #TESTZEILE
        #print("/Users/houben/PhD/ogs5/executable/ogs5 "+str(cwd)+str(dirpath[1:])+"/"+str(filename[:-4])+
        #      " >"+str(cwd)+str(dirpath[1:])+"/"+str(filename[:-4])+"_"+str(time[:-9])+".log")
        os.system("/Users/houben/PhD/ogs5/executable/ogs5 "+str(cwd)+str(dirpath[1:])+"/"+str(filename[:-4])+
              " >"+str(cwd)+str(dirpath[1:])+"/"+str(filename[:-4])+"_"+str(time[:-9])+".log")

print("ogs runs finished")

