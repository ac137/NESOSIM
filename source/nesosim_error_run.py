# run nesosim multiple times and save output


import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.interpolate import griddata
import os
import sys
sys.path.append('../')
import utils as cF
import io_helpers as io
from config import forcing_save_path,figure_path,oib_data_path,model_save_path
import NESOSIM


WAT = 5# 2par use default wat


# generate arrays of uncertainty 

OIB_STATUS = 'detailed'

if OIB_STATUS == 'detailed':
#oib no clim
# oib detailed
	central_wpf = 2.049653558530976e-06
	central_llf = 4.005362127700446e-07
	central_wpf_sigma = 2.6e-07
	central_llf_sigma = 4.9e-08

# make this directory if it doesn't exist
model_save_path = '/users/jk/19/acabaj/nesosim_uncert_output_oib_{}/'.format(OIB_STATUS)

# generate random distributions

# number of iterations/points to generate
N = 10 # just do 10 for now

wpf_vals = np.random.normal(central_wpf,central_wpf_sigma,N)
llf_vals = np.random.normal(central_llf,central_llf_sigma,N)



yearS=2010
yearE=2011
# these start and end variables are for loading data; load the whole year
month1 = 0
day1 = 0
month2 = 11 #is this the indexing used?, ie would this be december
day2 = 30 # would this be the 31st? I think so

# model parameters for input
precipVar='ERA5'
windVar='ERA5'
concVar='CDR'
driftVar='OSISAF'
dxStr='100km'
extraStr='v11'
dx = 100000 # for log-likelihood


# load forcings for nesosim
forcing_io_path=forcing_save_path+dxStr+'/'
print('loading input data')
forcing_dict = io.load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcing_io_path)
print('finished loading input')


# run the model

print('running {} iterations for year {}'.format(N, yearS))
# do a year with coincident ICESat-2 data (later); 
# for now just go with calibration year
year1=yearS

# if I do this for 5 years instead of just one...
# this part of the process is quite fast thankfully
# but the main model part is slow
for i in range(N):

	print('iteration ',i)

	WPF = wpf_vals[i]
	LLF = llf_vals[i]

	month1=month_start-1 # 8=September
	day1=day_start-1

	year2=year1+1
	month2=3 # 4=May
	day2=29

	date_start = pd.to_datetime('{}{:02d}{:02d}'.format(year1,month1+1,day1+1))


	# Get time period info
	_, _, _, dateOut=cF.getDays(year1, month1, day1, year2, month2, day2)
	#totalOutStr=''+folderStr+'-'+dateOut

	# run nesosim (not sure if all the additional variables are needed)

	# save the data; run saveData=1
	# lotsa things hardcoded here but this'll do for now
	budgets = NESOSIM.main(year1=year1, month1=month1, day1=day1, year2=year1+1, month2=month2, day2=day2,
    outPathT=model_save_path, 
    forcingPathT=forcing_save_path, 
    figPathT=figure_path+'Model/',
    precipVar='ERA5', windVar='ERA5', driftVar='OSISAF', concVar='CDR', 
    icVar='ERA5', densityTypeT='variable', extraStr='v11', outStr='mcmc', IC=2, 
    windPackFactorT=WPF, windPackThreshT=WAT, leadLossFactorT=LLF,
    dynamicsInc=1, leadlossInc=1, windpackInc=1, atmlossInc=1, saveData=1, plotBudgets=0, plotdaily=0,
    scaleCS=True, dx=dx,returnBudget=1, forcingVals=forcing_dict)