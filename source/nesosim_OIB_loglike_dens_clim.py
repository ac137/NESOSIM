# log-likelihood for NESOSIM vs. OIB; density climatology

import numpy as np
#from pylab import *
#from scipy.io import netcdf
import numpy.ma as ma
# from matplotlib import rc
from glob import glob
from scipy.interpolate import griddata
import sys
sys.path.append('../')
import utils as cF
# import xarray as xr
import pandas as pd
import os
import scipy.stats as st


from calendar import monthrange

import NESOSIM


from config import forcing_save_path,figure_path,oib_data_path,model_save_path


# need to modify nesosim so that it doesn't save files but instead 
# returns the snow depth data

# write oib function to get oib data separately

def get_OIB_and_mask(dx, yearT, depthBudget, date_start):#, days_ds, diff_ds):
	"""Grid all the OIB data and correlate"""

	# rewrite this to load the OIB data only once?

	xptsGMall=[]
	yptsGMall=[]
	snowDepthMMall=[]
	snowOIBMall=[]
	densMMall = []


	# nesosim depth which is then indexed per day
	iceConc = np.array(depthBudget['iceConc'])
	snowData = (depthBudget['snowDepth'][:, 0] + depthBudget['snowDepth'][:, 1])/iceConc
	# this is now indexable by day
	snowData = np.ma.masked_where(iceConc<0.15, snowData)

	# lonG, latG, xptsG, yptsG, nx, ny = cF.getGrid(, dx)
	xptsG, yptsG, latG, lonG, proj = cF.create_grid(dxRes=dx)


	dxStr=str(int(dx/1000))+'km'
	# region_maskG=load(forcingPath+'/Grid/regionMaskG'+dxStr)
	anc_data_pathT = '../anc_data/'
	forcingPath = forcing_save_path

	region_mask, xptsI, yptsI = cF.get_region_mask_pyproj(anc_data_pathT, proj, xypts_return=1)
	region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')

	folderPath=forcingPath+'/OIB/{}binned/{}/MEDIAN/'.format(dxStr,yearT)
	days_list = os.listdir(folderPath)
	
	for file_day in days_list:

#		print('File:', file_day)
		day_val = (pd.to_datetime(file_day[:8])-date_start).days

		try:
			# print(os.path.join(folderPath,file_day))
			snowDepthOIB=np.load(os.path.join(folderPath,file_day),allow_pickle=True)
			# transpose (when using old OIB files)
			# don't need transpose for new oib files (median); commenting out
		#	snowDepthOIB = snowDepthOIB.T

		except:
			continue

#		print('Num points in day:', np.size(snowDepthOIB))
		if (np.size(snowDepthOIB)==0):
			#nodata
			continue
		# if (day_val>259):
		# 	#beyond May 1st
		# 	continue

		# snow depth from NESOSIM output (budget); select single day
		snowDepthM = snowData[day_val]
		# density from NESOSIM - select single day
		# masking
		maskDay=np.zeros((xptsG.shape[0], xptsG.shape[1]))
		maskDay[snowDepthM.mask]=1
		# exclude NaN for OIB
		#maskDay[where(np.isnan(snowDepthOIB))]=1

		# exclude all NaN for all days

		# here is probably a good place to exclude the difference as well
		# what to do for days where there's some products missing?

		# get OIB difference for corresponding day
		# diff_val_day = diff_ds[i,:,:].values

		# maskDay[where(np.isnan(diff_val_day))]=1

		# where difference is greater than 10 cm
		# maskDay[where(diff_val_day>0.1)]=1
		maskDay[np.where(np.isnan(snowDepthOIB))]=1
		# maskDay[where(np.isnan(snowDepthOIB))]
		maskDay[np.where(snowDepthOIB<=0.04)]=1
		# get rid of masking based on nesosim output; varies per params and affects mcmc optimization
#		maskDay[np.where(snowDepthM<=0.04)]=1

		maskDay[np.where(snowDepthOIB>0.8)]=1
#		maskDay[np.where(snowDepthM>0.8)]=1

		maskDay[np.where(region_maskG>8.2)]=1

		# apply mask (once calculated)
		# does this masking mask everything to OIB? -> double-check
		snowDepthMM= snowDepthM[maskDay<1]
		xptsGM= xptsG[maskDay<1]
		yptsGM= yptsG[maskDay<1]
		snowDepthOIBM= snowDepthOIB[maskDay<1]

		xptsGMall.extend(xptsGM)
		yptsGMall.extend(yptsGM)
		# CONVERT TO CENTIMETERS!
		snowDepthMMall.extend(snowDepthMM*100.)
		snowOIBMall.extend(snowDepthOIBM*100.)

	return snowOIBMall, snowDepthMMall

# def get_density_clim():
# 	# calculate density climatology month by month

def calc_loglike(model_depth, obs_depth, model_dens, obs_dens, uncert_depth, uncert_dens, weight_dens=1):
	'''log likelihood for normal distribution
	based on likelihood function exp (0.5*sum((model-obs)^2/uncert^2))
	calculating for density and depth
	weight_dens: for weighting density by the number of depth observations'''
	depth_loglike = -0.5*np.sum((model_depth - obs_depth)**2/uncert_depth**2)
	dens_loglike = -0.5*weight_dens*np.sum((model_dens - obs_dens)**2/uncert_dens**2)
	return depth_loglike + dens_loglike


def calc_dens_monthly_means(depthBudget, date_start):
	'''monthly mean densities given budget (dataarray) and start date (datetime)
	returns an array of monthly mean densities (single value for each month)
	as a pandas dataframe
	'''
	# be sure to mask with ice concentration!
	iceConc = depthBudget['iceConc']
	density = depthBudget['density']

	density = density.where(iceConc<0.15)
	# create date range
	dates = pd.date_range(start=date_start,periods=density.shape[0])
	# assign as index to density
	density = density.assign_coords(time=dates)
	# basin average density
	density = density.mean(axis=(1,2))
	# this will be labelled by the last day of the month
	mon_means = density.resample(time='M').mean()

	return mon_means.to_dataframe()


def calc_clim(df):
	'''calculate monthly climatology from dataframe df
	outputs months in numerical order'''
	grp = df.groupby(df.index.month)
	return grp.mean()#, grp.std()


# various model parameters etc. - are these needed?
precipVar='ERA5'
reanalysis=precipVar
CSstr='CSscaled'
#CSstr=''
windVar='ERA5'
driftVar='OSISAF'
concVar='CDR'
densityTypeT='variable'
IC=2
dynamicsInc=1
windpackInc=1
leadlossInc=1
atmlossInc=0
windPackFactorT=5.8e-7
windPackThreshT=5
#leadLossFactorT=1.16e-6
leadLossFactorT=2.9e-7
dx = 100000

day_start = 1
month_start = 9

#TODO copy files over (these could be saved as csv instead of h5)
# mind your units; multiply by 1000 to convert from g/cm^3 to kg/m^3
station_dens_clim = pd.read_hdf('drifting_station_monthly_clim.h5',key='clim')['Mean Density']*1000
station_dens_std = pd.read_hdf('drifting_station_monthly_clim.h5',key='std')['Mean Density']*1000

def main(params, uncert):
	'''log-likelihood calculation for NESOSIM vs. OIB
	steps:
	- set up date variables
	- iterate over years:
		- run nesosim
		- mask to oib/select days
		- calculate log-likelihood and other stats
	params: parameters (wind packing, blowing snow, and wind action threshold); 
		may be constant or varying
	uncert: obs uncertainty estimate on OIB

	returns:
	logp: log-likelihood probability for nesosim vs. oib
	stats: list of [Pearson correlation, RMSE, mean error, standard deviation
	(OIB vs. NESOSIM), NESOSIM standard deviation, OIB standard deviation]
	'''


	# default wpf 5.8e-7
	# default llf 2.9e-7 

	# passing params as [wpf, llf]
	WPF = params[0]
	LLF = params[1]
	WAT = params[2]

	# windPackFactorT, leadLossFactorT = params
	# folderStr=precipVar+CSstr+'sf'+windVar+'winds'+driftVar+'drifts'+concVar+'sic'+'rho'+densityTypeT+'_IC'+str(IC)+'_DYN'+str(dynamicsInc)+'_WP'+str(windpackInc)+'_LL'+str(leadlossInc)+'_AL'+str(atmlossInc)+'_WPF'+str(windPackFactorT)+'_WPT'+str(windPackThreshT)+'_LLF'+str(leadLossFactorT)+'-'+dxStr+extraStr+outStr

	# run the model here maybe? and then grab the values somehow...
	startYear=2010
	endYear=2015
	numYears=endYear-startYear+1
	years=[str(year) for year in range(startYear, endYear+1)]
	years.append('All years')
	snowDepthOIBAll=[]
	snowDepthMMAll=[]
	densMMAll = []

	# if I do this for 5 years instead of just one...
	# this part of the process is quite fast thankfully
	# but the main model part is slow
	for year1 in range(startYear, endYear):
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

		budgets = NESOSIM.main(year1=year1, month1=month1, day1=day1, year2=year1+1, month2=month2, day2=day2,
	    outPathT=model_save_path, 
	    forcingPathT=forcing_save_path, 
	    figPathT=figure_path+'Model/',
	    precipVar='ERA5', windVar='ERA5', driftVar='OSISAF', concVar='CDR', 
	    icVar='ERA5', densityTypeT='variable', extraStr='v11', outStr='mcmc', IC=2, 
	    windPackFactorT=WPF, windPackThreshT=WAT, leadLossFactorT=LLF,
	    dynamicsInc=1, leadlossInc=1, windpackInc=1, atmlossInc=1, saveData=0, plotBudgets=0, plotdaily=0,
	    scaleCS=True, dx=dx,returnBudget=1)


		# get depth by year for given product & density
		# note: snowdepthoibyr/snowdepthmmyr should just be 1-d arrays with
		# only the valid values at this point, not 2d arrays with nan
		snowDepthOIByr, snowDepthMMyr = get_OIB_and_mask(dx, year2, budgets, date_start)
		
		dens_monthly_mean = calc_dens_monthly_means(budgets, date_start)

		snowDepthOIBAll.extend(snowDepthOIByr)
		snowDepthMMAll.extend(snowDepthMMyr)
		densMMAll.append(dens_monthly_mean)



	# collect density obs as well



	# calculate the log-likelihood
	# convert to arrays
	snowDepthMMAll = np.array(snowDepthMMAll)
	snowDepthOIBAll = np.array(snowDepthOIBAll)

	# number of obs, for weighting; assume there's no nan since those
	# are masked out, so can just use len
	# obs_count = np.count_nonzero(~np.isnan(snowDepthOIBAll))
	obs_count = len(snowDepthOIBAll)
	print('the observation count is {}'.format(obs_count))

	# stitch density dataframes together
	densMMAll = pd.concat(densMMAll)
	clim_dens = calc_clim(densMMAll)
	densMMAll = clim_dens.values
	# snow density arrays from station data; DS for 'drifting station'

	# selecting values here is a bit redundant (could just store immediately in these variables)
	# but I'll leave this for now

	# grab index directly from calc_clim output before taking values
	# so that density indices are consistent
	densDSAll = station_dens_clim.loc[clim_dens.index].values
	densUncert = station_dens_std.loc[clim_dens.index].values

	# weight for densities so they have same contribution as depth obs
	# is equal weighting too much?
	weight_factor = 0.05 # factor to scale weight down
	dens_weight = weight_factor*obs_count/len(densMMAll)
#	dens_weight = 4 # just multiply by 2
	print('the density weight is {}'.format(dens_weight))

	log_p = calc_loglike(snowDepthMMAll, snowDepthOIBAll, densMMAll, densDSAll, uncert, densUncert, dens_weight)

	# calculate other statistics for reference
	# linear fit with pearson correlation
	trend, sig, r_a, intercept = cF.correlateVars(snowDepthMMAll,snowDepthOIBAll)
	# rmse
	rmse=np.sqrt(np.mean((np.array(snowDepthMMAll)-np.array(snowDepthOIBAll))**2))
	# mean error
	merr=np.mean(np.array(snowDepthMMAll)-np.array(snowDepthOIBAll))
	# standard deviation nesosim vs. oib
	std=np.std(np.array(snowDepthMMAll)-merr-np.array(snowDepthOIBAll))
	# nesosim standard deviation
	std_n = np.std(snowDepthMMAll)
	# oib standard deviation
	std_o = np.std(snowDepthOIBAll)

	return log_p, [r_a, rmse, merr, std, std_n, std_o]
