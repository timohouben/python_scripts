## evaluate_results.py
## identify_misfits()
- add check for [inf] at covariance matrix

## multi_psd.py
- save covariance matrix as numbers!!


## single_psd_hetero.py
- SCRIPT HAT TO RUN FOR EVERY FOLDER INDIPENDENTLY SO THAT ERROS DON'T INFLUENCE THE FOLLOWING FOLDERS

## spectral_analysis_mpi.py
- ERROR: Optimals parameters not found!


## shh_anlytical
- code the complex shh_anlytical_2015 and test it against shh_anlytical_2013
- write the fitting function


## General stuff
- equalize the matplotlib stuff (e.g. fonts, latex or not etc)
- structure the module loads
- Include a File for the OGS setup, for observation point, their distance to the ricer and the name!!!! or improve GET_OBS (not generic enough)
- Set up a generic script to perform the SA on a single folder but parallel on EVE (partly done)
- Don't use the index of time series, use the actual time!

## ToBeDone
- Find an appropriate measure to compare input and output parameters.
- imporove labeling for legend in plot_errors_vs_loc
