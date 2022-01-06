# calculate error from output produced by nesosim_error_run.py
# i.e. from ensemble of nesosim runs, get uncertainty


import numpy as np
import pandas as pd
import xarray as xr
import os

model_save_path = '/users/jk/19/acabaj/nesosim_uncert_output_oib_{}/100km/'.format(OIB_STATUS)

# typical file format: /users/jk/19/acabaj/nesosim_uncert_output_oib_detailed/100km/ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_AL1_WPF2.2673462045700356e-06_WPT5_LLF4.567476109493221e-07-100kmv11mcmc/final

dir_list = os.listdir(model_save_path)

# load data

# collect into list because there's 

final_data = []

# can use xr open_mfdataset with concat_dim
# if I need to number them, can refer to this:
# https://stackoverflow.com/questions/42574705/specify-concat-dim-for-xarray-open-mfdataset

model_data = xr.open_mfdataset('/users/jk/19/acabaj/nesosim_uncert_output_oib_detailed/100km/ERA*/final/*.nc',concat_dim='iteration_number')



# calculate mean and uncertainty of snow depth and density


print(model_data)