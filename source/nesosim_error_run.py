# nesosim_error_run.py: Generate a "parameter ensemble" of NESOSIM runs. Given
# distributions of NESOSIM model free parameters, select n samples of parameters
# from those distributions and run NESOSIM n times. Model uncertainty due to
# parameter uncertainty can then be calculated using nesosim_error_calc.py

# by Alex Cabaj; adapted from code by Alek Petty


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


forcing_save_path = '/users/jk/18/acabaj/NESOSIM/forcings_full/forcings/'



WAT = 5 # wind action threshold; default value


# generate arrays of uncertainty 

# Select whether to use the OIB-clim ("averaged") or daily-gridded ("detailed")
# configuration

# OIB_STATUS = 'averaged'
OIB_STATUS = 'detailed'

USE_COV = True # Use covariance (from covariance matrix of MCMC output)
USE_IC = False # use initial conditions; for testing initial condition factor

if OIB_STATUS == 'detailed':
	# 'detailed' meaning using gridded oib in likelihood function
	# i.e. comparing each oib grid square to a respective nesosim grid square

	# example for distributions
	central_wpf = 2.0504155592128743e-06
	central_llf = 4.0059442776163867e-07
	central_wpf_sigma = 2.8e-07
	central_llf_sigma = 4.9e-08
	# full covariance

	# covariance matrix, as used in publication
	cov = np.array([[9.67094599e-14, 1.37744084e-14], [1.37744084e-14, 2.81528717e-15]])

elif OIB_STATUS == 'averaged':

	# 'averaged' meaning using oib climatology in likelihood function
	# i.e. comparing the oib monthly regionally-averaged climatology to the
	# nesosim monthly regionally-averaged climatology (over a region spanning
	# the oib study region)

	if USE_IC:

		# for evaluating initial conditions (i.e. 3 parameter test)
		central_wpf = 2.3450925692135826e-06
		central_llf = 1.5380250062998322e-07
		central_icf = 0.5312831368932197

		cov = np.array([[ 2.06599304e-13,  4.84971256e-15, -3.68061026e-09],
	       [ 4.84971256e-15,  5.16078775e-15,  1.01706829e-10],
	       [-3.68061026e-09,  1.01706829e-10,  7.66086674e-04]])
	else:

		central_wpf = 1.7284668037515452e-06
		central_llf = 1.2174787315012357e-07
		central_wpf_sigma = 2.6e-07
		central_llf_sigma = 6.6e-08

		cov = np.array([[6.84908674e-14, 1.86872558e-16],[1.86872558e-16, 4.39607346e-15]])


# extra string for filename formatting
EXTRA_FMT = '40_years_final_5k'
# EXTRA_FMT = 'final_5k_2018_2019'
if USE_IC:
	EXTRA_FMT += 'with_ic_loglike'

# generate random distributions

# number of iterations/ensemble members to generate
N = 100


if USE_COV:

	# use covariance; generate distribution from joint distribution
	if USE_IC:
		means = [central_wpf, central_llf, central_icf]
	else:
		means = [central_wpf, central_llf]

	joint_dist =  np.random.multivariate_normal(means, cov, N)

	wpf_vals = joint_dist[:,0]
	llf_vals = joint_dist[:,1]

	if USE_IC:
		icf_vals = joint_dist[:,2]

	# append to model save path
	EXTRA_FMT += '_cov'

else:
	# don't use covariance; independent distributions

	wpf_vals = np.random.normal(central_wpf,central_wpf_sigma,N)
	llf_vals = np.random.normal(central_llf,central_llf_sigma,N)


# directory where files are saved: may need creation if it does not exist
model_save_path = '/users/jk/20/acabaj/nesosim_uncert_output_oib_{}{}/'.format(OIB_STATUS, EXTRA_FMT)



yearS=1980
yearE=2020
# these start and end variables are for loading data; load the whole year
month1 = 0
day1 = 0
month2 = 11 
day2 = 30 

# these are for starting the actual model itself
day_start = 1 # first day; gets subtracted later for day1
month_start = 9 # september; gets subtracted later for month1

# model parameters for input
precipVar='ERA5'
windVar='ERA5'
concVar='CDR'
driftVar='NSIDCv4'
#driftVar='OSISAF'
dxStr='100km'
extraStr='v11'
dx = 100000 # for log-likelihood


# load forcings for nesosim
# for 1980-2020 this would be ~5 gb loaded into ram; comment out and use default
# model config if this is too much
forcing_io_path=forcing_save_path+dxStr+'/'
print('loading input data')
forcing_dict = io.load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcing_io_path)
print('finished loading input')


# run the model

print('running {} iterations starting at year {}'.format(N, yearS))

for i in range(N):

	print('iteration ',i)

	WPF = wpf_vals[i]
	LLF = llf_vals[i]

	if USE_IC:
		# select initial condition factor
		ICF = icf_vals[i]
	else:
		# otherwise just use default
		ICF = 1

	# loop here for multiple years

	for y in range(yearS, yearE):

		if y == 1987:
			# skip this year due to missing data
			continue

		print('year {}'.format(y))
		year1 = y

		month1=month_start-1 # 8=September
		day1=day_start-1

		year2=year1+1
		month2=3 # 4=May
		day2=29

		date_start = pd.to_datetime('{}{:02d}{:02d}'.format(year1,month1+1,day1+1))


		# Get time period info
		_, _, _, dateOut=cF.getDays(year1, month1, day1, year2, month2, day2)


		# run NESOSIM

		# save the data; run saveData=1
		budgets = NESOSIM.main(year1=year1, month1=month1, day1=day1, year2=year1+1, month2=month2, day2=day2,
	    outPathT=model_save_path, 
	    forcingPathT=forcing_save_path, 
	    figPathT=figure_path+'Model/',
	    precipVar='ERA5', windVar='ERA5', driftVar='NSIDCv4', concVar='CDR', 
	    icVar='ERA5', densityTypeT='variable', extraStr='v11', outStr='mcmc', IC=2, 
	    windPackFactorT=WPF, windPackThreshT=WAT, leadLossFactorT=LLF,
	    dynamicsInc=1, leadlossInc=1, windpackInc=1, atmlossInc=1, saveData=1, plotBudgets=0, plotdaily=0,
	    scaleCS=True, dx=dx,returnBudget=1, forcingVals=forcing_dict, ICfactor=ICF)
