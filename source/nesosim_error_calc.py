# Calculate error statistics from output produced by nesosim_error_run.py;
# i.e. given an ensemble of NESOSIM runs, calculate the mean and standard 
# deviation. Produces an estimate of model uncertainty due to parameter
# uncertainty.

# Note: intput/output file paths are hardcoded; adjust as necessary

# by Alex Cabaj; using output from NESOSIM.
# NESOSIM was originally developed by Alek Petty and is available at
# https://github.com/akpetty/NESOSIM

import numpy as np
import pandas as pd
import xarray as xr
import os
from dask.diagnostics import ProgressBar


years_list = np.arange(1988,2020)

for current_year in years_list:
	print(current_year)

	# formatting string for filename
	EXTRA_FMT = '40_years_final_5k_cov'

	# Select whether to use the OIB-clim ("averaged") or daily-gridded ("detailed")
	# configuration
	OIB_STATUS = 'detailed'
	# OIB_STATUS = 'averaged'

	# load all ensemble data simultaneously; create additional dimension based 
	# on iteration number
	model_data = xr.open_mfdataset('/users/jk/20/acabaj/nesosim_uncert_output_oib_{}{}/100km/ERA*/final/NESOSIMv11_0109{}-3004{}.nc'.format(OIB_STATUS,EXTRA_FMT,current_year,current_year+1),combine='nested',concat_dim='iteration_number')


	# calculate mean and standard deviation of snow depth and density


	print('calculating mean')
	with ProgressBar():
		mean_vals = model_data.mean(dim='iteration_number').compute()
	#print(mean_vals)

	print('calculating standard deviation')
	with ProgressBar():
		uncert_vals = model_data.std(dim='iteration_number').compute()
	# print(uncert_vals)

	n_iter = 100 # number of iterations, actually only needed for file name

	print('saving data for {}', current_year)
	mean_vals.to_netcdf('/users/jk/19/acabaj/nesosim_uncert_output_oib_{}{}/{}{}mean_{}_iter_final_{}.nc'.format(OIB_STATUS,EXTRA_FMT, OIB_STATUS, EXTRA_FMT,n_iter, current_year))
	uncert_vals.to_netcdf('/users/jk/19/acabaj/nesosim_uncert_output_oib_{}{}/{}{}uncert_{}_iter_final_{}.nc'.format(OIB_STATUS,EXTRA_FMT, OIB_STATUS, EXTRA_FMT,n_iter, current_year))


