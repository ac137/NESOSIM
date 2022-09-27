

# 3-parameter mcmc varying initial conditions, wpf, and llf,
# with initial conditions as another term in log-likelihood

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


# use density in mcmc constraints
USE_DENS = False
USE_DENS_CLIM = True

# # is this sort of control flow for import statements reasonable? hopefully
# if USE_DENS:
# 	import nesosim_OIB_loglike_dens as loglike
# elif USE_DENS_CLIM:
# 	import nesosim_oib_loglike_dens_clim_io as loglike
# else:
# 	import nesosim_OIB_loglike as loglike

def get_grids(dx):
	xptsG, yptsG, latG, lonG, proj = cF.create_grid(dxRes=dx)


	dxStr=str(int(dx/1000))+'km'
	# region_maskG=load(forcingPath+'/Grid/regionMaskG'+dxStr)
	anc_data_pathT = '../anc_data/'
	forcingPath = forcing_save_path

	region_mask, xptsI, yptsI = cF.get_region_mask_pyproj(anc_data_pathT, proj, xypts_return=1)
	region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')
	return region_maskG, xptsG, yptsG


def preload_oib(dxStr, startYear, endYear):
	'''preload operation icebridge data; returns a dict of dict
	structured as
	year
	-> day number/oib data
	-> -> values for each day
	'''
	# starts in 2010, ends in 2015

	year_dict = {}


	for year1 in range(startYear, endYear):
		month1=month_start-1 # 8=September
		day1=day_start-1


		yearT=year1+1
		forcingPath = forcing_save_path
		date_start = pd.to_datetime('{}{:02d}{:02d}'.format(year1,month1+1,day1+1))

		folderPath=forcingPath+'/OIB/{}binned/{}/MEDIAN/'.format(dxStr,yearT)
		days_list = os.listdir(folderPath)

		# dictionary for collecting data for a single year
		d = {}

		# lists to collect days and oib data
		days = []
		oibdata = []
		
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

			# now we have the daily data, append to lists
			days.append(day_val)
			oibdata.append(snowDepthOIB)

		# have all daily data, populate sub-dictionary
		d['days']=days
		d['OIB']=oibdata

		# put in main dictionary
		year_dict[yearT] = d

	return year_dict



def get_OIB_and_mask(dx, yearT, depthBudget, date_start, region_maskG, xptsG, yptsG, oib_dict):#, days_ds, diff_ds):
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
	# xptsG, yptsG, latG, lonG, proj = cF.create_grid(dxRes=dx)


	# dxStr=str(int(dx/1000))+'km'
	# # region_maskG=load(forcingPath+'/Grid/regionMaskG'+dxStr)
	# anc_data_pathT = '../anc_data/'
	forcingPath = forcing_save_path

	# region_mask, xptsI, yptsI = cF.get_region_mask_pyproj(anc_data_pathT, proj, xypts_return=1)
	# region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')

	# folderPath=forcingPath+'/OIB/{}binned/{}/MEDIAN/'.format(dxStr,yearT)
	# days_list = os.listdir(folderPath)

	days_list = oib_dict[yearT]['days']
	oib_list = oib_dict[yearT]['OIB']
	
	for i, day_val in enumerate(days_list):

#		print('File:', file_day)
# 		day_val = (pd.to_datetime(file_day[:8])-date_start).days

# 		try:
# 			# print(os.path.join(folderPath,file_day))
# 			snowDepthOIB=np.load(os.path.join(folderPath,file_day),allow_pickle=True)
# 			# transpose (when using old OIB files)
# 			# don't need transpose for new oib files (median); commenting out
# 		#	snowDepthOIB = snowDepthOIB.T

# 		except:
# 			continue

# #		print('Num points in day:', np.size(snowDepthOIB))
# 		if (np.size(snowDepthOIB)==0):
# 			#nodata
# 			continue
		# if (day_val>259):
		# 	#beyond May 1st
		# 	continue
		snowDepthOIB = oib_list[i]

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

# def calc_loglike(model_depth, obs_depth, model_dens, obs_dens, uncert_depth, uncert_dens, weight_dens=1):
def calc_loglike(model_depth, obs_depth, model_dens, obs_dens, model_depth_clim, 
				obs_depth_clim, uncert_depth, uncert_dens, uncert_depth_clim,
				model_day_zero_depth, default_day_zero_depth, 
				day_zero_uncert_factor, weight_dens=1, weight_depth=1):
	
	'''log likelihood for normal distribution
	based on likelihood function exp (0.5*sum((model-obs)^2/uncert^2))
	calculating for density and depth
	weight_dens: for weighting density by the number of depth observations'''
	depth_loglike = -0.5*np.sum((model_depth - obs_depth)**2/uncert_depth**2)
	dens_loglike = -0.5*weight_dens*np.sum((model_dens - obs_dens)**2/uncert_dens**2)
	depth_clim_loglike = -0.5*weight_depth*np.sum((model_depth_clim-obs_depth_clim)**2/uncert_depth_clim**2)
	
	day_zero_uncert = default_day_zero_depth*day_zero_uncert_factor
	day_zero_loglike = -0.5*np.sum((model_day_zero_depth - default_day_zero_depth)**2/day_zero_uncert**2)

	# add log-likelihoods together
	return depth_loglike + dens_loglike + depth_clim_loglike + day_zero_loglike


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


def get_depth_day_zero(depthBudget):
	'''return snow depth on zeroth day of mcmc run
	returns nxn array where n = size of model grid'''

	# may need to check indices here
	# select first day
	budget_day_zero = depthBudget.isel(time=0)
	iceConc = np.array(budget_day_zero['iceConc'])
	depth = (budget_day_zero['snowDepth'].isel(lyrs=0) + budget_day_zero['snowDepth'].isel(lyrs=1))#/iceConc
	# don't need to divide by iceconc here because it's being compared to a quantity that isn't being divided by day 0 
	#print(depth)
#	depth = depth.where(iceConc<0.15)
	#print(depth['snowDepth'])
	depth_vals = np.ma.masked_where(iceConc<0.15, depth)
	depth.values = depth_vals

	return depth_vals


def calc_depth_monthly_means(depthBudget, date_start):
	'''monthly mean densities given budget (dataarray) and start date (datetime)
	returns an array of monthly mean depths (single value for each month)
	as a pandas dataframe
	'''
	iceConc = np.array(depthBudget['iceConc'])
	depth = (depthBudget['snowDepth'][:, 0] + depthBudget['snowDepth'][:, 1])/iceConc
	#print(depth)
#	depth = depth.where(iceConc<0.15)
	#print(depth['snowDepth'])
	depth_vals = np.ma.masked_where(iceConc<0.15, depth)
	depth.values = depth_vals
#	depth = depth.where(~np.isinf(depth.values))
#	depth.where(np.isinf(depth.values))=np.nan
#	depth[np.isinf(depth.values)]=np.nan
	#print(depth)
#	depth.values = 
	# create date range
	dates = pd.date_range(start=date_start,periods=depth.shape[0])
	# assign as index to depth
	depth = depth.assign_coords(time=dates)
	# basin average depth
	depth = depth.mean(axis=(1,2),skipna=True)
	#print(depth)
	# this will be labelled by the last day of the month
	mon_means = depth.resample(time='M').mean()

	return mon_means.to_dataframe()


def calc_depth_mean_oib_region(depthBudget, date_start, indices):
	# how is depth budget indexing formatted

	# first select depth by index
	# indices: axis0 min, max; axis1 min, max
	

	# more efficient to index first and then calculate depth budget;
	# also more finicky

	# slice depth budget to region coincident with oib
	budget_regional = depthBudget.isel(x=slice(indices[0],indices[1]),y=slice(indices[2],indices[3]))
	# now the budget is sliced, calculate monthly means

	mon_means_regional = calc_depth_monthly_means(depthBudget, date_start)

	return mon_means_regional

def calc_clim(df):
	'''calculate monthly climatology from dataframe df
	outputs months in numerical order'''
	grp = df.groupby(df.index.month)
	return grp.mean()#, grp.std()

def loglike(params, uncert, forcings, weight_factor=None):
	'''log-likelihood calculation for NESOSIM vs. OIB
	steps:
	- set up date variables
	- iterate over years:
		- run nesosim
		- mask to oib/select days
		- calculate log-likelihood and other stats
	params: parameters (wind packing, blowing snow, and init. cond. factor); 
		may be constant or varying
	uncert: obs uncertainty estimate on OIB
	forcings: NESOSIM forcings (preloaded as dictionary)
	weight_factor: optional: weight in terms of the number of OIB observations

	returns:
	logp: log-likelihood probability for nesosim vs. oib
	stats: list of [Pearson correlation, RMSE, mean error, standard deviation
	(OIB vs. NESOSIM), NESOSIM standard deviation, OIB standard deviation]
	'''


	# default wpf 5.8e-7
	# default llf 2.9e-7 
	indices = [31,59,21,52] # oib depth mean region indices; hardcoding over here for now


	# passing params as [wpf, llf]
	WPF = params[0]
	LLF = params[1]
	# WAT = params[2]
	WAT = 5 # default wind action threshold
	ICF = params[2] # initial condition factor

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
	depthMMAll = [] # depth monthly means
	depth_mean_oib_region_all = [] #depth monthly means for oib region
	depths_day_0 = []

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

		# lotsa things hardcoded here but this'll do for now
		budgets = NESOSIM.main(year1=year1, month1=month1, day1=day1, year2=year1+1, month2=month2, day2=day2,
	    outPathT=model_save_path, 
	    forcingPathT=forcing_save_path, 
	    figPathT=figure_path+'Model/',
	    precipVar='ERA5', windVar='ERA5', driftVar='OSISAF', concVar='CDR', 
	    icVar='ERA5', densityTypeT='variable', extraStr='v11', outStr='mcmc', IC=2, 
	    windPackFactorT=WPF, windPackThreshT=WAT, leadLossFactorT=LLF,
	    dynamicsInc=1, leadlossInc=1, windpackInc=1, atmlossInc=1, saveData=0, plotBudgets=0, plotdaily=0,
	    scaleCS=True, dx=dx,returnBudget=1, forcingVals=forcings, ICfactor=ICF)


		# get depth by year for given product & density
		# note: snowdepthoibyr/snowdepthmmyr should just be 1-d arrays with
		# only the valid values at this point, not 2d arrays with nan

		if CLIM_OIB:
			# using oib climatology; calculating monthly mean in oib region
			depth_monthly_mean_oib_region = calc_depth_mean_oib_region(budgets, date_start, indices)
			depth_mean_oib_region_all.append(depth_monthly_mean_oib_region)
		# else:
		# not using oib region but still want these for stats

		snowDepthOIByr, snowDepthMMyr = get_OIB_and_mask(dx, year2, budgets, date_start, region_maskG, xptsG, yptsG, oib_dict)
		snowDepthOIBAll.extend(snowDepthOIByr) # nb. extend method is less efficient but I don't think it's the main bottleneck here
		snowDepthMMAll.extend(snowDepthMMyr)

		dens_monthly_mean = calc_dens_monthly_means(budgets, date_start)
		depth_monthly_mean = calc_depth_monthly_means(budgets, date_start)

		densMMAll.append(dens_monthly_mean)
		depthMMAll.append(depth_monthly_mean)
		depth_on_first_day = get_depth_day_zero(budgets)

		depths_day_0.append(depth_on_first_day)




	# collect density obs as well



	# calculate the log-likelihood
	# convert to arrays

	if CLIM_OIB:
		print('using oib climatology')
		# stitch oib-region nesosim depth mean
		depth_mean_oib_region_all = pd.concat(depth_mean_oib_region_all)
		clim_depth_nesosim_oib_region = calc_clim(depth_mean_oib_region_all)
		# obs_count=2 #just 2 months; putting this in for now but not so necessary; clean up weighting code later?

	# still want these for stats:
	snowDepthMMAll = np.array(snowDepthMMAll)
	snowDepthOIBAll = np.array(snowDepthOIBAll)

	# number of obs, for weighting; assume there's no nan since those
	# are masked out, so can just use len
	# obs_count = np.count_nonzero(~np.isnan(snowDepthOIBAll))
	obs_count = len(snowDepthOIBAll)
	print('the oib observation count is {}'.format(obs_count))

	# stitch density dataframes together & calculate climatology
	densMMAll = pd.concat(densMMAll)
	clim_dens = calc_clim(densMMAll)
	densMMAll = clim_dens.values

	# stitch depth dataframes together & calculate NESOSIM depth climatology
	depthMMAll = pd.concat(depthMMAll)
	clim_depth = calc_clim(depthMMAll)
	depthMMAll = clim_depth.values
	print(depthMMAll)

	# snow density arrays from station data; DS for 'drifting station'

	# selecting values here is a bit redundant (could just store immediately in these variables)
	# but I'll leave this for now

	# grab index directly from calc_clim output before taking values
	# so that density indices are consistent
	densDSAll = station_dens_clim.loc[clim_dens.index].values
	densUncert = station_dens_std.loc[clim_dens.index].values

	# also grab the buoy depths here somewhere

	buoyDepthAll = buoy_depth_clim.loc[clim_depth.index].values
	buoyUncert = buoy_depth_std.loc[clim_depth.index].values
	# index nesosim depth by oib months
	# ie subset the nesosim depth clim by the months that the oib
	# clim exists (because the latter is the limiting factor here)
	
	if CLIM_OIB:

		clim_depth_nesosim_oib_region = clim_depth_nesosim_oib_region.loc[oib_depth_clim.index].values


	# stack day-zero depths

	depths_day_0 = np.stack(depths_day_0)

	# weight for densities so they have same contribution as depth obs
	# is equal weighting too much?
	if weight_factor:
	# weight_factor = 0.05 # factor to scale weight down
		dens_weight = weight_factor*obs_count/len(densMMAll)
		depth_weight = dens_weight
	else:
		dens_weight = 1.
		depth_weight = dens_weight
#	dens_weight = 4 # just multiply by 2

	# just set buoy depth weight to 1 for now
	# trying now with multiples of 4
	dens_weight = 1
	depth_weight = 1

	print('the density weight is {} and the buoy depth weight is {}'.format(dens_weight, depth_weight))


	if CLIM_OIB:
		log_p = calc_loglike(clim_depth_nesosim_oib_region, 
							oib_depth_clim.values, densMMAll, densDSAll, 
							depthMMAll, buoyDepthAll, uncert, densUncert, 
							buoyUncert,  depths_day_0, initial_condition_values, 
							initial_condition_uncertainty_factor, dens_weight, depth_weight
							)

# 		initial_condition_values = np.stack(initial_condition_values)


# # initial condition uncertainty factor = 50%
# initial_condition_uncertainty_factor = 0.5
	else:
		log_p = calc_loglike(snowDepthMMAll, snowDepthOIBAll, densMMAll, 
							densDSAll, depthMMAll, buoyDepthAll, uncert, 
							densUncert, buoyUncert, depths_day_0, 
							initial_condition_values, 
							initial_condition_uncertainty_factor, dens_weight, 
							depth_weight)


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

def write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls):
	'''write output with to an hdf file at location fname'''
	'''note: will overwrite files!'''

	stat_headings = ['r','rmse','merr','std','std_n','std_o']
	valid_df = pd.DataFrame(np.array(stats_list), columns=stat_headings)
	par_arr = np.array(par_list)
	# save all the paramters (iterate over in case of variable length)
	for i in range(len(par_names)):
		valid_df[par_names[i]] = par_arr[:,i]

	valid_df['loglike'] = loglike_list

	rejected_df = pd.DataFrame(np.array(rejected_stats), columns=stat_headings)
	rej_par_arr = np.array(rejected_pars)
	for i in range(len(par_names)):
		rejected_df[par_names[i]] = rej_par_arr[:,i]
	rejected_df['loglike'] = rejected_lls

	valid_df.to_hdf(fname, key='valid')
	rejected_df.to_hdf(fname, key='rejected')
	meta_df.to_hdf(fname, key='meta')



# variables for log-like

# load station density climatologies
# mind your units; multiply by 1000 to convert from g/cm^3 to kg/m^3

station_dens_clim = pd.read_hdf('drifting_station_monthly_clim.h5',key='clim')['Mean Density']*1000
station_dens_std = pd.read_hdf('drifting_station_monthly_clim.h5',key='std')['Mean Density']*1000

# do I have to convert the units??? these are in m and so is nesosim so prob not
buoy_depth_clim = pd.read_hdf('buoy_monthly_clim.h5',key='clim')['Snow Depth (m)']
buoy_depth_std = pd.read_hdf('buoy_monthly_clim.h5',key='std')['Snow Depth (m)']

# oib clim (just has march and april)

oib_depth_clim = pd.read_hdf('oib_monthly_clim.h5',key='clim')['daily mean']
oib_depth_std = pd.read_hdf('oib_monthly_clim.h5',key='std')['daily mean']

# load initial conditions from 2010-2015
initial_condition_values = []

for y_current in range(2010,2015): # double-check year range
	ic_year = np.load(forcing_save_path + '100km/InitialConditions/ERA5/ICsnow{}-100kmv11'.format(y_current), allow_pickle=True)
	initial_condition_values.append(ic_year)
	# do these need scaling for units???

# stack initial condition values
initial_condition_values = np.stack(initial_condition_values)


# initial condition uncertainty factor = 50%
initial_condition_uncertainty_factor = 0.5

# seed for testing
#np.random.seed(42)

# default wpf 5.8e-7
# default llf 2.9e-7 ? different default for multiseason

ITER_MAX = 10000# start small for testing
#ITER_MAX = 5
UNCERT = 10 # obs uncertainty for log-likelihood (also can be used to tune)
# par_vals = [1., 1.] #initial parameter values

PAR_SIGMA = [1, 1, 0.1] # standard deviation for parameter distribution; can be separate per param
# should be multiplied by 1e-7, but can do that after calculating distribution

# step size determined based on param uncertainty (one per parameter)

# weighting 1x n_oib for both now
LOGLIKE_WEIGHT = 1 # this isn't being used for now


# if true, use OIB climatology; 'averaged oib'
# be sure to also change DENS_STR to match
#CLIM_OIB = True
CLIM_OIB = False


if USE_DENS:
	DENS_STR = '_density'
elif USE_DENS_CLIM:
	DENS_STR = '_density_clim'
else:
	DENS_STR = ''

# weighting is now specified by passing argument to loglike main
# for density clim loglike
# using half-weighting (cf loglike file) so change filename
# DENS_STR+= '_w0.05'
#DENS_STR += '_3par_fixed_ic_station_buoy_oib_averaged_defaultweights'
DENS_STR += '_3par_fixed_ic_station_buoy_oib_detailed_defaultweights'


# try over both wpf and lead loss, now
# order here is [wpf, llf]
#par_vals = np.array([5.8e-7, 2.9e-7])
#par_vals = np.array([5.8e-7, 1.45e-7, 5.])
# par_vals = np.array([5.8e-7, 2.9e-7, 5.])
#continue from previous mcmc with last accepted value
# par_vals = np.array([4.12616198947269e-06, 8.416761649739341e-07, 0.19611063365133324])
# par_vals = np.array([6.220783261481277e-06, 1.2792313785323853e-06, 0.1546572899704643])

# wpf, llf, icf (initial condition factor, default is 1)
par_vals = np.array([5.8e-7, 2.9e-7, 1.])


PARS_INIT = par_vals.copy()
par_names = ['wind packing', 'blowing snow','initial condition factor']

metadata_headings = ['N_iter','uncert','prior_p1','prior_p2', 'prior_p3','sigma_p1','sigma_p2', 'sigma_p3','oib_prod']
metadata_values = [[ITER_MAX, UNCERT, par_vals[0], par_vals[1], par_vals[2],PAR_SIGMA[0], PAR_SIGMA[1], PAR_SIGMA[2], 'MEDIAN']]
meta_df = pd.DataFrame(metadata_values, columns=metadata_headings)


NPARS = len(par_vals)


# load nesosim input

yearS=2010
yearE=2015
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


# get grids (hardcoded variables for now I guess)

region_maskG, xptsG, yptsG = get_grids(dx)


# vars for log-likelihood; day and month start for nesosim
# n.b. loglike references some global vars and his hardcoded
# maybe re-work later
# also hardcoded into oib preload, oops
day_start = 1
month_start = 9


# preload oib data

oib_dict = preload_oib(dxStr, yearS, yearE)

forcing_io_path=forcing_save_path+dxStr+'/'


print('loading input data')
forcing_dict = io.load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcing_io_path)
print('finished loading input')



# pass nesosim input to log-likelihood



print('calculating initial log-likelihood')
p0, stats_0 = loglike(par_vals, UNCERT, forcing_dict, weight_factor=LOGLIKE_WEIGHT) # initial likelihood function
print ('initial setup: params {}, log-likelihood: {}'.format(par_vals, p0))
print('r, rmse, merr, std, std_n, std_o')
print(stats_0)

par_list = [par_vals] # now an nxm list (m = # of pars)
loglike_list = [p0]
stats_list = [stats_0] # collect rmse and r also, etc.
#var_cond_list=[]
#diff_list=[]
rejected_pars = []
rejected_lls = []
rejected_stats = []


# first just try metropolis (don't reject proposed values of params)

# steps to take
# np.randon.normal(mean, sigma, shape); sigma can be an array

# maybe change this later to not pre-calculate steps so that
# this doesn't take up space in memory
step_vals = np.random.normal(0, PAR_SIGMA, (ITER_MAX, NPARS))#*1e-7

# scale to appropriate value
step_vals[:,0] *= 1e-7 # scale wind packing
step_vals[:,1] *= 1e-7 # scale blowing snow
# don't scale IC factor


# reshape this if the number of params changes
# reject any new parameter less than 0


# open files to store parameters
acceptance_count = 0

for i in range(ITER_MAX):
	print('iterating')
	# random perturbation to step; adjust choice here
	rand_val = step_vals[i]

	# adjust parameters (not checking param distribution for now)
	par_new = par_vals + rand_val

	print('new parameter ', par_new)

	# if any of the parameters are less than zero, don't step there;
	# conversely, if none of the parameters are less than zero, proceed

	# to check param distribution, check the difference of par_new and par_vals
	# not doing this for now

	if (par_new < 0).any() == False:
		print('calculating new log-likelihood')
		# calculate new log-likelihood
		p, stats = loglike(par_new, UNCERT, forcing_dict, weight_factor=LOGLIKE_WEIGHT)

		# accept/reject; double-check this with mcmc code
		# in log space, p/q becomes p - q, so check difference here
		# checking with respect to uniform distribution
		var_cond = np.log(np.random.rand())
#		var_cond_list.append(var_cond)
#		diff = p-p0
#		diff_list.append(diff)
		if p-p0 > var_cond:
			# accept value
			print('accepted value')
			acceptance_count += 1
#			print('acceptance rate: {}/{} = {}'.format(acceptance_count,i+1,acceptance_count/float(i+1)))
			par_vals = par_new
			p0 = p
			# append to list/ possibly save these to disk so that interrupting the
			# process doesn't lose all the info?
			print('parameters ', par_vals)
			par_list.append(par_vals)
			loglike_list.append(p0)
			stats_list.append(stats)
		else:
			print('rejected value')
			rejected_pars.append(par_new)
			rejected_lls.append(p)
			rejected_stats.append(stats)

		print('acceptance rate: {}/{} = {}'.format(acceptance_count,i+1,acceptance_count/float(i+1)))
	if i%1000 == 0 and i > 0:
		# save output every 1k iterations just in case
		print('Writing output for {} iterations...'.format(i))
		# use ITER_MAX to overwrite here, i to create separate files (more disk space but safer)
		fname = 'mcmc_output_i{}_u_{}_p0_{}_{}_{}_s0_{}_{}_{}_{}noseed.h5'.format(i,UNCERT,PARS_INIT[0],PARS_INIT[1],PARS_INIT[2],PAR_SIGMA[0],PAR_SIGMA[1],PAR_SIGMA[2],DENS_STR)
		write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls)



#TODO: more elegant filename formatting (format arrays so I don't have to write strings in)
# save final output to file
fname = 'mcmc_output_i{}_u_{}_p0_{}_{}_{}_s0_{}_{}_{}_{}_with_ic_loglike_noseed.h5'.format(ITER_MAX,UNCERT,PARS_INIT[0],PARS_INIT[1],PARS_INIT[2],PAR_SIGMA[0],PAR_SIGMA[1],PAR_SIGMA[2],DENS_STR)
print(ITER_MAX)
print(fname)
write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls)

