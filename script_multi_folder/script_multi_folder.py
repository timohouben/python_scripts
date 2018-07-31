#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:27:11 2018

@author: houben

script to run a script in all subdirectories
When you execute this script you will be asked to enter a path and file name to execute in all subdirectories of your curent working directoy.
"""
import os
import os.path
import datetime
import shutil
#import subprocess


cwd_start = os.getcwd()

pathname_script = raw_input("Enter path and name of script: ")
extension = raw_input("Enter name of extension for which will be searched (with .): ")

#subprocess.call(str(pathname_script))



for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith(str(extension))]:
        #print os.path.join(dirpath, filename)
        time = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
        print("Start time: "+str(time))
        print("Script Run : "+str(dirpath[1:]))
        path_temp = str(cwd_start)+str(dirpath[1:])
        print(path_temp)
        shutil.copy2(str(pathname_script), str(path_temp)+'/TEMP.py')
        cwd_temp = os.getcwd()
        os.chdir(str(path_temp))
        os.system(str(path_temp)+'/TEMP.py')
        os.remove(str(path_temp)+'/TEMP.py')
        os.chdir(str(cwd_temp))
        #os.system("/Users/houben/PhD/ogs5/executable/ogs5 "+str(cwd)+str(dirpath[1:])+"/"+str(filename[:-4])+
         #     " >"+str(cwd)+str(dirpath[1:])+"/"+str(filename[:-4])+"_"+str(time[:-9])+".log")

print("Done.")
