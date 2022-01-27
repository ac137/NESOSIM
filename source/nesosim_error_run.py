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

# OIB_STATUS = 'averaged'
OIB_STATUS = 'detailed'

USE_COV = True

if OIB_STATUS == 'detailed':
	# 'detailed' meaning using gridded oib in likelihood function
	# i.e. comparing each oib grid square to a respective nesosim grid square

	# 
	# central_wpf = 2.049653558530976e-06
	# central_llf = 4.005362127700446e-07
	# central_wpf_sigma = 2.6e-07
	# central_llf_sigma = 4.9e-08

	# # now with added covariance!
	# cov = np.array([[6.71091118e-14, 9.04186813e-15],[9.04186813e-15, 2.35878636e-15]])
	
	# next 5k iterations (last 800 iter avg)
	# central_wpf = 2.0504155592128743e-06
	# central_llf = 4.0059442776163867e-07
	# central_wpf_sigma = 3.1e-07
	# central_llf_sigma = 5.3e-08
	# cov = np.array([[9.67094599e-14, 1.37744084e-14], [1.37744084e-14, 2.81528717e-15]])
	# 

	# last 5k iterations (sigma and cov calculated using all last 5k iterations)

	central_wpf = 2.0504155592128743e-06
	central_llf = 4.0059442776163867e-07
	central_wpf_sigma = 2.8e-07
	central_llf_sigma = 4.9e-08

	cov = np.array([[7.85960133e-14, 1.10931288e-14], [1.10931288e-14, 2.39994177e-15]])

elif OIB_STATUS == 'averaged':
	# 'averaged' meaning using oib climatology in likelihood function
	# i.e. comparing the oib monthly regionally-averaged climatology to the
	# nesosim monthly regionally-averaged climatology (over a region spanning
	# the oib study region)

	# central_wpf = 1.6321262995790887e-06
	# central_llf = 1.1584399852081886e-07
	# central_wpf_sigma = 2.3e-07
	# central_llf_sigma = 5.9e-08

	# cov = np.array([[ 5.16310381e-14, -6.14010167e-16], [-6.14010167e-16,  3.47444174e-15]])

	# next 5k iterations (last 800 iter avg)

	# central_wpf = 1.7284668037515452e-06
	# central_llf = 1.2174787315012357e-07
	# central_wpf_sigma = 2.7e-07
	# central_llf_sigma = 6.8e-08

	# cov = np.array([[7.10691349e-14, 1.97960231e-15],[1.97960231e-15, 4.63158306e-15]])

	# parameters from last 5k iterations (sigma and cov calculated using all last 5k iterations)
	# these should be the final ones used
	central_wpf = 1.7284668037515452e-06
	central_llf = 1.2174787315012357e-07
	central_wpf_sigma = 2.6e-07
	central_llf_sigma = 6.6e-08

	cov = np.array([[6.84908674e-14, 1.86872558e-16],[1.86872558e-16, 4.39607346e-15]])


EXTRA_FMT = 'final_5k'
# make this directory if it doesn't exist

# generate random distributions

# number of iterations/points to generate
N = 100# going for 100 total initially


if USE_COV:

	# use covariance; generate distribution from joint distribution
	means = [central_wpf, central_llf]

	joint_dist =  np.random.multivariate_normal(means, cov, N)

	wpf_vals = joint_dist[:,0]
	llf_vals = joint_dist[:,1]

	# append to model save path
	EXTRA_FMT.append('_cov')

else:
	# don't use covariance; independent distributions

	wpf_vals = np.random.normal(central_wpf,central_wpf_sigma,N)
	llf_vals = np.random.normal(central_llf,central_llf_sigma,N)


model_save_path = '/users/jk/19/acabaj/nesosim_uncert_output_oib_{}{}/'.format(OIB_STATUS, EXTRA_FMT)



yearS=2010
yearE=2011
# these start and end variables are for loading data; load the whole year
month1 = 0
day1 = 0
month2 = 11 #is this the indexing used?, ie would this be december
day2 = 30 # would this be the 31st? I think so

# these are for starting the actual model itself
day_start = 1 # first day; gets subtracted later for day1
month_start = 9 # september; gets subtracted later for month1

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
