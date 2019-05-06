"""
script to execute the extract_timeseries() from transect_plot.py.
"""

from transect_plot import extract_timeseries

try:
    path_to_project = sys.argv[1]
except IndexError:
    path_to_project = input("Insert path to project: ")

try:
    which = sys.argv[2]
    process = sys.argv[3]
    rfd = sys.argv[4]
    plot = sys.argv[5]
except: IndexErro:
    which = "mean"
    rfd = "1"
    process="GROUNDWATER_FLOW"
    plot = True

extract_timeseries(path_to_project, which=mean, process=process, rfd=rfd)

if plot == True:
    from transect_plot import plot_head_timeseries_vs_recharge
    plot_head_timeseries_vs_recharge(path_to_project, which=which)
