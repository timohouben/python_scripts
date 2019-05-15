#!/usr/bin/env pythonw
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import os
import numpy as np
import re

time = []
x_coord = []
h = []

time2_lst = []
averageH_lst = []
Q_lst = []
Kup_lst = []
P_lst = []

time2 = 0
averageH = 0
Q = 0
Kup = 0
P = 0

filename1 = "H.OUT"
filename2 = "AquiferScale.OUT"

#### test if steady or transient flow is selected
steady = False
steady_test_file = open("Dupuitflow.in", "r")
lst = []
for line in steady_test_file:
    lst.append(line)
if (
    str(lst[4]) == "STEADY\r\n"
    or str(lst[4]) == "Steady\r\n"
    or str(lst[4]) == "steady\r\n"
):
    steady = True

#### extracts the number of observation points and number of time steps
n_locations = sum(1 for line in open("OutputLocations.in", "r"))
n_times = sum(1 for line in open("OutputTimes.in", "r"))
h_array = np.zeros([n_times, n_locations])
time_array = np.zeros([n_times, n_locations])
x_coord_array = np.zeros([n_times, n_locations])

#### open file, create lst without whitespace and split in between
def getlist(filename):
    data_file = open(filename, "r")
    lst = []
    for string in data_file:
        line = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", string)
        lst.append(line)
    del lst[0:2]
    data_file.close()
    return lst


#### distribute values of time, x_coord and h on different variables
def split(data):
    for i in range(0, len(data)):
        time.append(float(data[i][0]))
        x_coord.append(float(data[i][1]))
        h.append(float(data[i][2]))


#### creates arrays from the lists to plot with wireframe for H.OUT data
#### creates seperate lists for each value from the lists to plot with line for AquiferScale.OUT data
def makearray_H(n_locations, n_times):
    if steady == False:
        i = 0
        k = 0
        while i < n_times:
            j = 0
            line = float(data2[i][0])
            time2_lst.append(line)
            line = float(data2[i][1])
            averageH_lst.append(line)
            line = float(data2[i][2])
            Q_lst.append(line)
            line = float(data2[i][3])
            Kup_lst.append(line)
            line = float(data2[i][4])
            P_lst.append(line)
            while j < n_locations:
                h_array[i, j] = h[k]
                time_array[i, j] = time[k]
                x_coord_array[i, j] = x_coord[k]
                k = k + 1
                j = j + 1
            i = i + 1
        time2_lst.pop(0)  # pop 1st element with time = 0
        averageH_lst.pop(0)
        Q_lst.pop(0)
        P_lst.pop(0)
        Kup_lst.pop(0)
    else:
        global time2
        time2 = float(data2[0][0])
        global averageH
        averageH = float(data2[0][1])
        global Q
        Q = float(data2[0][2])
        global Kup
        Kup = float(data2[0][3])
        global P
        P = float(data2[0][4])


####    3D wireframe plot in one direction
# Y = observation points
# Z = head
# X = time
def plot():
    if steady == False:
        Y = x_coord_array
        Z = h_array
        X = time_array
        # fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(15, 10), subplot_kw={'projection': '3d'})
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle("results GWModel run de Rooij 2012")

        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax1 = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
        ax.set_title("head over time and location")
        ax.view_init(40, 130)
        ax.set_xlabel("time")
        ax.set_ylabel("location")
        ax.set_zlabel("head")

        ax = fig.add_subplot(2, 4, 3)
        ax2 = ax.plot(time2_lst, averageH_lst)
        ax.set_title("average head")
        ax.set_xlabel("time")
        ax.set_ylabel("head [m]")

        ax = fig.add_subplot(2, 4, 4)
        ax3 = ax.plot(time2_lst, P_lst)
        ax.set_title("perturbation measure")
        ax.set_xlabel("time")
        ax.set_ylabel("perturbation")

        ax = fig.add_subplot(2, 4, 7)
        ax4 = ax.plot(time2_lst, Q_lst)
        ax.set_title("flux")
        ax.set_xlabel("time")
        ax.set_ylabel("flux")

        ax = fig.add_subplot(2, 4, 8)
        ax5 = ax.plot(time2_lst, Kup_lst)
        ax.set_title("aquifer-scale hydraulic conductivity")
        ax.set_xlabel("time")
        ax.set_ylabel("conductivity")
        fig.savefig("plots.png")
        plt.show()

    else:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(x_coord, h)
        ax.set_title("steady state head")
        ax.set_xlabel("location")
        ax.set_ylabel("head [m]")
        ax.annotate(
            "average H = " + str(averageH),
            xy=(2, 1),
            xytext=(max(x_coord) - 30, max(h)),
        )
        ax.annotate("Q = " + str(Q), xy=(2, 1), xytext=(max(x_coord) - 30, max(h) - 20))
        ax.annotate(
            "Kup = " + str(Kup), xy=(2, 1), xytext=(max(x_coord) - 30, max(h) - 40)
        )
        ax.annotate("P = " + str(P), xy=(2, 1), xytext=(max(x_coord) - 30, max(h) - 60))
        fig.savefig("plots.png")
        plt.show()

    return fig


""" ax2.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
    ax2.set_title("View B")
    ax2.view_init(30, 90)
    plt.xlabel('time')
    plt.ylabel('location')
    ax2.set_xlabel('time')
    ax2.set_ylabel('location')
    ax2.set_zlabel('head')
   
    ax3.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
    ax3.set_title("View C")
    ax3.view_init(40, 160)
    plt.xlabel('time')
    plt.ylabel('location')
    ax3.set_xlabel('time')
    ax3.set_ylabel('location')
    ax3.set_zlabel('head')
  
    ax4.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
    ax4.set_title("View D")
    ax4.view_init(30, 50)
    plt.xlabel('time')
    plt.ylabel('location')
    ax4.set_xlabel('time')
    ax4.set_ylabel('location')
    ax4.set_zlabel('head')
"""

#### save plots
# def saveplot(fig):
#    path = r'plots'
#    if not os.path.isdir(path):
#        os.makedirs(path)
#        fig.savefig('plots/image.png')
#        print("Folder 'plots' created and plots saved")
#    else:
#        fig.savefig('plots/image.png')
#        print("Plots saved!")

#### execute script
data1 = getlist(filename1)
data2 = getlist(filename2)
split(data1)
makearray_H(n_locations, n_times)
fig = plot()
# saveplot(fig)
