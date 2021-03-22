# log-likelihood for NESOSIM vs. OIB

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

def get_OIB_and_mask(dx, yearT, depthBudget,date_start):#, days_ds, diff_ds):
	"""Grid all the OIB data and correlate"""

	# rewrite this to load the OIB data only once?

	xptsGMall=[]
	yptsGMall=[]
	snowDepthMMall=[]
	snowOIBMall=[]


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

	folderPath=forcingPath+'/OIB/{}binned/{}/'.format(dxStr,yearT)
	days_list = os.listdir(folderPath)
	
	for file_day in days_list:

#		print('File:', file_day)
		day_val = (pd.to_datetime(file_day[:8])-date_start).days

		try:
			# print(os.path.join(folderPath,file_day))
			snowDepthOIB=np.load(os.path.join(folderPath,file_day),allow_pickle=True)
			# transpose (when using old OIB files)
			snowDepthOIB = snowDepthOIB.T

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

def calc_loglike(model, obs, uncert):
	# log likelihood for normal distribution
	# based on likelihood function exp (0.5*sum((model-obs)^2/uncert^2))
	return -0.5*np.sum((model - obs)**2/uncert**2)


# various model parameters etc.
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


def main(params, uncert):
	'''log-likelihood calculation for NESOSIM vs. OIB
	steps:
	- set up date variables
	- iterate over years:
		- run nesosim
		- mask to oib/select days
		- calculate log-likelihood and other stats
	params: parameters to be varied; currently just varying lead loss factor
	(single parameter changes)
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
	    windPackFactorT=WPF, windPackThreshT=5, leadLossFactorT=LLF,
	    dynamicsInc=1, leadlossInc=1, windpackInc=1, atmlossInc=1, saveData=0, plotBudgets=0, plotdaily=0,
	    scaleCS=True, dx=dx,returnBudget=1)


		# get depth by year for given product
		snowDepthOIByr, snowDepthMMyr= get_OIB_and_mask(dx, year2, budgets,date_start)
		snowDepthOIBAll.extend(snowDepthOIByr)
		snowDepthMMAll.extend(snowDepthMMyr)

	# calculate the log-likelihood
	snowDepthMMAll = np.array(snowDepthMMAll)
	snowDepthOIBAll = np.array(snowDepthOIBAll)
	log_p = calc_loglike(snowDepthMMAll, snowDepthOIBAll, uncert)

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
