# i/o optimized nesosim mcmc script (n-pars) with log-likelihood
# in same script (avoid some import overhread?)
# or would this be less efficient since it's not compiled? 
# can try and check


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


# use density in mcmc constraints; this just impacts the file name currently
# need to manually adjust calc_loglike
USE_DENS = False
USE_DENS_CLIM = True


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

		snowDepthOIB = oib_list[i]

		# snow depth from NESOSIM output (budget); select single day
		snowDepthM = snowData[day_val]
		# density from NESOSIM - select single day
		# masking
		maskDay=np.zeros((xptsG.shape[0], xptsG.shape[1]))
		maskDay[snowDepthM.mask]=1

		# where difference is greater than 10 cm
		maskDay[np.where(np.isnan(snowDepthOIB))]=1
		maskDay[np.where(snowDepthOIB<=0.04)]=1
		# get rid of masking based on nesosim output; varies per params and affects mcmc optimization

		maskDay[np.where(snowDepthOIB>0.8)]=1
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

def calc_loglike(model_depth, obs_depth, model_dens, obs_dens, model_depth_clim, obs_depth_clim, uncert_depth, uncert_dens, uncert_depth_clim, weight_dens=1, weight_depth=1):
	'''log likelihood for normal distribution
	based on likelihood function exp (0.5*sum((model-obs)^2/uncert^2))
	calculating for density and depth
	weight_dens: for weighting density by the number of depth observations'''
#	print(model_depth)
#	print(obs_depth)
	depth_loglike = -0.5*np.sum((model_depth - obs_depth)**2/uncert_depth**2)
#	depth_loglike = 0
	dens_loglike = -0.5*weight_dens*np.sum((model_dens - obs_dens)**2/uncert_dens**2)
	# dens_loglike = 0
	depth_clim_loglike = -0.5*weight_depth*np.sum((model_depth_clim-obs_depth_clim)**2/uncert_depth_clim**2)
	return depth_loglike + dens_loglike + depth_clim_loglike


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
	params: parameters (wind packing, blowing snow, and wind action threshold); 
		may be constant or varying
	uncert: obs uncertainty estimate on OIB
	forcings: NESOSIM forcings (preloaded as dictionary)
	weight_factor: optional: weight in terms of the number of OIB observations

	returns:
	logp: log-likelihood probability for nesosim vs. oib
	stats: list of [Pearson correlation, RMSE, mean error, standard deviation
	(OIB vs. NESOSIM), NESOSIM standard deviation, OIB standard deviation]

	note: references a few global variables assigned outside of function:
	- oib depth climatology (if applicable); oib_depth_clim
	- buoy depth climatology; buoy_depth_clim
	- drifting station density climatology; station_dens_clim
	'''


	# default wpf 5.8e-7
	# default llf 2.9e-7 
	indices = [31,59,21,52] # oib depth mean region indices; hardcoding over here for now

	# passing params as [wpf, llf]
	WPF = params[0]
	LLF = params[1]
#	WAT = params[2]
	WAT = 5# 2par use default wat

	# variables for mcmc model run; the constants could be moved outside the 
	# function (so as not to hardcode) but leaving them here for now
	startYear=2010
	endYear=2015
	numYears=endYear-startYear+1
	years=[str(year) for year in range(startYear, endYear+1)]
	years.append('All years')
	snowDepthOIBAll=[]
	snowDepthMMAll=[]
	densMMAll = [] # density monthly means
	depthMMAll = [] # depth monthly means
	depth_mean_oib_region_all = [] #depth monthly means for oib region

	# loop over years to run NESOSIM
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

		# run nesosim (many values hardcoded; may need changing for different configs)
		budgets = NESOSIM.main(year1=year1, month1=month1, day1=day1, year2=year1+1, month2=month2, day2=day2,
	    outPathT=model_save_path, 
	    forcingPathT=forcing_save_path, 
	    figPathT=figure_path+'Model/',
	    precipVar='ERA5', windVar='ERA5', driftVar='OSISAF', concVar='CDR', 
	    icVar='ERA5', densityTypeT='variable', extraStr='v11', outStr='mcmc', IC=2, 
	    windPackFactorT=WPF, windPackThreshT=WAT, leadLossFactorT=LLF,
	    dynamicsInc=1, leadlossInc=1, windpackInc=1, atmlossInc=1, saveData=0, plotBudgets=0, plotdaily=0,
	    scaleCS=True, dx=dx,returnBudget=1, forcingVals=forcings)


		# get depth by year for given product & density
		# note: snowdepthoibyr/snowdepthmmyr should just be 1-d arrays with
		# only the valid values at this point, not 2d arrays with nan
		
		if CLIM_OIB:
			# using oib climatology; calculating monthly mean in oib region
			depth_monthly_mean_oib_region = calc_depth_mean_oib_region(budgets, date_start, indices)
			depth_mean_oib_region_all.append(depth_monthly_mean_oib_region)
		
		# even if not using gridded OIB, still want these for stats maybe
		snowDepthOIByr, snowDepthMMyr = get_OIB_and_mask(dx, year2, budgets, date_start, region_maskG, xptsG, yptsG, oib_dict)
		snowDepthOIBAll.extend(snowDepthOIByr) 
		snowDepthMMAll.extend(snowDepthMMyr)
		# n.b. extend is less efficient than pre-allocating full array and 
		# assigning values, but likely not the main bottleneck here

		# calculate density and depth monthly means for whole region 
		# (for station and buoy comparsons, respectively)
		dens_monthly_mean = calc_dens_monthly_means(budgets, date_start)
		depth_monthly_mean = calc_depth_monthly_means(budgets, date_start)

		# append to lists where monthly mean density and depth are being collected
		densMMAll.append(dens_monthly_mean)
		depthMMAll.append(depth_monthly_mean)
		


	# calculate the log-likelihood

	if CLIM_OIB:
		print('using oib climatology')
		# concatenate oib-region nesosim depth mean, calculate climatology
		# for comparison in likelihood function
		depth_mean_oib_region_all = pd.concat(depth_mean_oib_region_all)
		clim_depth_nesosim_oib_region = calc_clim(depth_mean_oib_region_all)

	# model & OIB snow depth
	snowDepthMMAll = np.array(snowDepthMMAll)
	snowDepthOIBAll = np.array(snowDepthOIBAll)

	# number of obs, for weighting; assume there's no nan since those
	# are masked out, so can just use len
	obs_count = len(snowDepthOIBAll)
	print('the oib observation count is {}'.format(obs_count))


	# concatenate density dataframes together & calculate climatology
	densMMAll = pd.concat(densMMAll)
	clim_dens = calc_clim(densMMAll)
	densMMAll = clim_dens.values

	# concatenate depth dataframes together & calculate NESOSIM depth climatology
	depthMMAll = pd.concat(depthMMAll)
	clim_depth = calc_clim(depthMMAll)
	depthMMAll = clim_depth.values
	print(depthMMAll)


	# snow density arrays from station data; DS for 'drifting station'
	# selecting values here is a bit redundant (could just store immediately in these variables)
	# but I'll leave this for now

	# get the indices lined up;
	# grab index directly from calc_clim output before taking values
	# so that density indices are consistent
	densDSAll = station_dens_clim.loc[clim_dens.index].values
	densUncert = station_dens_std.loc[clim_dens.index].values

	# also grab the buoy depths 
	buoyDepthAll = buoy_depth_clim.loc[clim_depth.index].values
	buoyUncert = buoy_depth_std.loc[clim_depth.index].values

	# index nesosim depth by oib months
	# ie subset the nesosim depth clim by the months that the oib
	# clim exists (because the latter is the limiting factor here)
	if CLIM_OIB:

		clim_depth_nesosim_oib_region = clim_depth_nesosim_oib_region.loc[oib_depth_clim.index].values

	# NOT USING WEIGHT FACTOR FOR NOW; SEE BELOW
	# weight for densities so they have same contribution as depth obs
	# is equal weighting too much?
	if weight_factor:
	# weight_factor = 0.05 # factor to scale weight down
		dens_weight = weight_factor*obs_count/len(densMMAll)
		depth_weight = dens_weight
	else:
		dens_weight = 1.
		depth_weight = dens_weight

	# NOT USING WEIGHT FACTOR FOR NOW
	dens_weight = 1
	depth_weight = 1


	print('the density weight is {} and the buoy depth weight is {}'.format(dens_weight, depth_weight))

	# calculate log-likelihood; pass different values depending if OIB-clim or OIB-daily-gridded
	if CLIM_OIB:
		log_p = calc_loglike(clim_depth_nesosim_oib_region, oib_depth_clim.values, densMMAll, densDSAll, depthMMAll, buoyDepthAll, uncert, densUncert, buoyUncert, dens_weight, depth_weight)
	else:
		log_p = calc_loglike(snowDepthMMAll, snowDepthOIBAll, densMMAll, densDSAll, depthMMAll, buoyDepthAll, uncert, densUncert, buoyUncert, dens_weight, depth_weight)

	# calculate other statistics for reference (wrt OIB)
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
	'''write MCMC output (accepted and rejected parameters with statistics)
	  to an hdf file at location fname.
	  note: will overwrite existing files!'''

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



# variables for log-likelihood (observations)

# load pre-calculated station density climatologies
# multiply by 1000 to convert from g/cm^3 to kg/m^3

station_dens_clim = pd.read_hdf('drifting_station_monthly_clim.h5',key='clim')['Mean Density']*1000
station_dens_std = pd.read_hdf('drifting_station_monthly_clim.h5',key='std')['Mean Density']*1000

# snow buoy depth climatologies
buoy_depth_clim = pd.read_hdf('buoy_monthly_clim.h5',key='clim')['Snow Depth (m)']
buoy_depth_std = pd.read_hdf('buoy_monthly_clim.h5',key='std')['Snow Depth (m)']

# oib monthly climatology (just has march and april)
oib_depth_clim = pd.read_hdf('oib_monthly_clim.h5',key='clim')['daily mean']
oib_depth_std = pd.read_hdf('oib_monthly_clim.h5',key='std')['daily mean']


# seed for testing
#np.random.seed(42)


# maximum number of iterations, start small for testing
ITER_MAX = 5000
UNCERT = 10 # obs uncertainty for log-likelihood (10 cm for OIB)


# prior parameter standard deviation; will be scaled down later
PAR_SIGMA = [1, 1] # 2 parameters

# step size determined based on param standard deviation (one per parameter)

# weight for different terms in log-likelihood; currently hardcoded to 1 in
# loglike(); this variable will not affect it
LOGLIKE_WEIGHT = 1

# if true, use OIB climatology; 'OIB-clim'/'oib-averaged'
CLIM_OIB = True

# leftover adjustments to filename
if USE_DENS:
	DENS_STR = '_density'
elif USE_DENS_CLIM:
	DENS_STR = '_density_clim'
else:
	DENS_STR = ''



# string added to filename to specify configuration
DENS_STR += '2par_io_final_averaged_w1_default_v1_default'

# parameter value array used in mcmc (this array is updated)
# try over both wind packing factor and blowing snow factor, now
# order here is [WP, BS]
par_vals = np.array([5.8e-7, 2.9e-7]) # prior values

#can also continue from previous mcmc with last accepted value

# par_vals = np.array([1.7026104191089884e-06, 1.0808249925065788e-07])

# initial (prior) parameter values
PARS_INIT = par_vals.copy()

# names of parameters used
par_names = ['wind packing', 'blowing snow']

# metadata for mcmc ouptut files
metadata_headings = ['N_iter','uncert','prior_p1','prior_p2', 'sigma_p1','sigma_p2', 'oib_prod']
metadata_values = [[ITER_MAX, UNCERT, par_vals[0], par_vals[1],PAR_SIGMA[0], PAR_SIGMA[1], 'MEDIAN']]

# metadata dataframe for saving
meta_df = pd.DataFrame(metadata_values, columns=metadata_headings)

# number of parameters
NPARS = len(par_vals)


# load nesosim input

yearS=2010
yearE=2015
# these start and end variables are for loading data; load the whole year
month1 = 0
day1 = 0
month2 = 11 #is this the indexing used?, ie would this be december
day2 = 30 # would this be the 31st? I think so

# model parameters for input, also referenced by loglike()
precipVar='ERA5'
windVar='ERA5'
concVar='CDR'
driftVar='OSISAF'
dxStr='100km'
extraStr='v11'
dx = 100000

# get grids for region mask for OIB comparison etc.
# these are also referenced by loglike()
region_maskG, xptsG, yptsG = get_grids(dx)


# vars for log-likelihood; day and month start for nesosim
# n.b. loglike() and the OIB preloading functions reference some global 
# variables without them being explicitly passed to the function. take caution
# when naming variables

day_start = 1
month_start = 9

# preload oib data; either for log-likelihood or for stats
oib_dict = preload_oib(dxStr, yearS, yearE)
forcing_io_path=forcing_save_path+dxStr+'/'

# load NESOSIM input data
print('loading input data')
forcing_dict = io.load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcing_io_path)
print('finished loading input')


# pass nesosim input to log-likelihood, calculate initial log-likelihood

print('calculating initial log-likelihood')
p0, stats_0 = loglike(par_vals, UNCERT, forcing_dict, weight_factor=LOGLIKE_WEIGHT) # initial likelihood function
print ('initial setup: params {}, log-likelihood: {}'.format(par_vals, p0))
print('r, rmse, merr, std, std_n, std_o')
print(stats_0)

# lists for collecting MCMC values/stats
par_list = [par_vals] # will become an n*m list of accepted parameters (m = # of pars)
loglike_list = [p0]
stats_list = [stats_0] # collect rmse and r also, etc.
rejected_pars = []
rejected_lls = []
rejected_stats = []


# metropolis mcmc

# pre-calculate all the MCMC steps from the prior
# (since the step isn't adaptive, this can be done all at once)
# not inputting 1e-7 as values since np.random.normal has difficulty with very
# large/small values. instead can scale afterwards
step_vals = np.random.normal(0, PAR_SIGMA, (ITER_MAX, NPARS))

# scale to appropriate value
step_vals[:,0] *= 1e-7 # scale wind packing
step_vals[:,1] *= 1e-7 # scale blowing snow
# reshape this if the number of params changes

# for calculating acceptance rate; may not actually be needed but leaving for now
acceptance_count = 0

for i in range(ITER_MAX):
	print('iterating')
	# select step size corresponding to iteration number (from pre-generated values)
	rand_val = step_vals[i]

	# update parameters
	par_new = par_vals + rand_val

	print('new parameter ', par_new)

	# if any of the parameters are less than zero, don't step there; 
	# (model won't run)
	# conversely, if none of the parameters are less than zero, proceed
	# in practice have not run into <0 parameters but keeping just in case

	if (par_new < 0).any() == False:
		print('calculating new log-likelihood')
		# calculate new log-likelihood
		p, stats = loglike(par_new, UNCERT, forcing_dict,weight_factor=LOGLIKE_WEIGHT)

		# acceptance/rejection step
		# checking with respect to log-uniform distribution
		var_cond = np.log(np.random.rand())

		if p-p0 > var_cond:
			# accept value
			print('accepted value')
			acceptance_count += 1
			par_vals = par_new
			p0 = p
			# append accepted values & stats
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
		# save intermediate output every 1k iterations just in case 
		print('Writing output for {} iterations...'.format(i))
		fname = 'mcmc_output_i{}_u_{}_p0_{}_{}_s0_{}_{}_{}noseed.h5'.format(i,UNCERT,PARS_INIT[0],PARS_INIT[1],PAR_SIGMA[0],PAR_SIGMA[1],DENS_STR)
		write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls)


#TODO: more elegant filename formatting (format arrays so I don't have to write strings in)
# save final output to file

fname = 'mcmc_output_i{}_u_{}_p0_{}_{}_s0_{}_{}_{}noseed.h5'.format(ITER_MAX,UNCERT,PARS_INIT[0],PARS_INIT[1],PAR_SIGMA[0],PAR_SIGMA[1],DENS_STR)

print(ITER_MAX)
print(fname)
write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls)

