# calculate error from output produced by nesosim_error_run.py
# i.e. from ensemble of nesosim runs, get uncertainty


import numpy as np
import pandas as pd
import xarray as xr
import os
from dask.diagnostics import ProgressBar


EXTRA_FMT = '_cov'
OIB_STATUS = 'detailed'
#OIB_STATUS = 'averaged'

#model_save_path = '/users/jk/19/acabaj/nesosim_uncert_output_oib_{}/100km/'.format(OIB_STATUS)

# typical file format: /users/jk/19/acabaj/nesosim_uncert_output_oib_detailed/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF2.2673462045700356e-06_WPT5_LLF4.567476109493221e-07-100kmv11mcmc/final

#dir_list = os.listdir(model_save_path)

# load data

# collect into list because there's 

#final_data = []

# can use xr open_mfdataset with concat_dim
# if I need to number them, can refer to this:
# https://stackoverflow.com/questions/42574705/specify-concat-dim-for-xarray-open-mfdataset

# model_data = xr.open_mfdataset('/users/jk/19/acabaj/nesosim_uncert_output_oib_{}_final/100km/ERA*/final/*.nc'.format(OIB_STATUS),combine='nested',concat_dim='iteration_number')

model_data = xr.open_mfdataset('/users/jk/19/acabaj/nesosim_uncert_output_oib_{}{}/100km/ERA*/final/*.nc'.format(OIB_STATUS,EXTRA_FMT),combine='nested',concat_dim='iteration_number')


# calculate mean and uncertainty of snow depth and density


print(model_data)

print('calculating mean')
with ProgressBar():
	mean_vals = model_data.mean(dim='iteration_number').compute()
#print(mean_vals)

print('calculating standard deviation')
with ProgressBar():
	uncert_vals = model_data.std(dim='iteration_number').compute()
print(uncert_vals)

n_iter = 100

print('saving data')
mean_vals.to_netcdf('/users/jk/19/acabaj/nesosim_uncert_output_oib_{}{}/{}{}mean_{}_iter_final.nc'.format(OIB_STATUS,EXTRA_FMT, OIB_STATUS, EXTRA_FMT,n_iter))
uncert_vals.to_netcdf('/users/jk/19/acabaj/nesosim_uncert_output_oib_{}{}/{}{}uncert_{}_iter_final.nc'.format(OIB_STATUS,EXTRA_FMT, OIB_STATUS, EXTRA_FMT,n_iter))


