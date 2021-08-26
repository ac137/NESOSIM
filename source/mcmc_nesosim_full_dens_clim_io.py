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

def get_OIB_and_mask(dx, yearT, depthBudget, date_start, region_maskG, xptsG, yptsG):#, days_ds, diff_ds):
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
	# forcingPath = forcing_save_path

	# region_mask, xptsI, yptsI = cF.get_region_mask_pyproj(anc_data_pathT, proj, xypts_return=1)
	# region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')

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

		# lotsa things hardcoded here but this'll do for now
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
		snowDepthOIByr, snowDepthMMyr = get_OIB_and_mask(dx, year2, budgets, date_start, region_maskG, xptsG, yptsG)
		
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
	if weight_factor:
	# weight_factor = 0.05 # factor to scale weight down
		dens_weight = weight_factor*obs_count/len(densMMAll)
	else:
		dens_weight = 1.
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



# seed for testing
#np.random.seed(42)

# default wpf 5.8e-7
# default llf 2.9e-7 ? different default for multiseason

ITER_MAX = 10# start small for testing
#ITER_MAX = 3
UNCERT = 5 # obs uncertainty for log-likelihood (also can be used to tune)
# par_vals = [1., 1.] #initial parameter values

PAR_SIGMA = [1, 1, 0.1] # standard deviation for parameter distribution; can be separate per param
# should be multiplied by 1e-7, but can do that after calculating distribution

# step size determined based on param uncertainty (one per parameter)




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
DENS_STR += '3par_io'

# try over both wpf and lead loss, now
# order here is [wpf, llf]
#par_vals = np.array([5.8e-7, 2.9e-7])
#par_vals = np.array([5.8e-7, 1.45e-7, 5.])
par_vals = np.array([5.8e-7, 2.9e-7, 5.])
#continue from previous mcmc with last accepted value
# par_vals = np.array([4.12616198947269e-06, 8.416761649739341e-07, 0.19611063365133324])
# par_vals = np.array([6.220783261481277e-06, 1.2792313785323853e-06, 0.1546572899704643])



PARS_INIT = par_vals.copy()
par_names = ['wind packing', 'blowing snow','wind action threshold']

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
day_start = 1
month_start = 9


forcing_io_path=forcing_save_path+dxStr+'/'


print('loading input data')
forcing_dict = io.load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcing_io_path)
print('finished loading input')



# pass nesosim input to log-likelihood



print('calculating initial log-likelihood')
p0, stats_0 = loglike(par_vals, UNCERT, forcing_dict) # initial likelihood function
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
# don't scale wind action threshold


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
		p, stats = loglike(par_new, UNCERT, forcing_dict)

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
fname = 'mcmc_output_i{}_u_{}_p0_{}_{}_{}_s0_{}_{}_{}_{}noseed.h5'.format(ITER_MAX,UNCERT,PARS_INIT[0],PARS_INIT[1],PARS_INIT[2],PAR_SIGMA[0],PAR_SIGMA[1],PAR_SIGMA[2],DENS_STR)
print(ITER_MAX)
print(fname)
write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls)

