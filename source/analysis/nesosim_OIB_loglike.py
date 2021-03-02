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


from config import forcing_save_path,figure_path,oib_data_path,model_save_path


# need to modify nesosim so that it doesn't save files but instead 
# returns the snow depth data

# write oib function to get oib data separately

# should this be the log likelihood or just the whole thing...


def get_OIB():
	# load oib data; do this only once!
	# do gridding separately
	# just use getOIBNESOSIM for now
	# 
	pass


def getOIBNESOSIM(dx, folderStr, totalOutStr, yearT, snowType, reanalysis):#, days_ds, diff_ds):
	"""Grid all the OIB data and correlate"""

	# rewrite this to load the OIB data only once?

	xptsGMall=[]
	yptsGMall=[]
	snowDepthMMall=[]
	snowOIBMall=[]

	# lonG, latG, xptsG, yptsG, nx, ny = cF.getGrid(, dx)
	xptsG, yptsG, latG, lonG, proj = cF.create_grid(dxRes=dx)




	dxStr=str(int(dx/1000))+'km'
	# region_maskG=load(forcingPath+'/Grid/regionMaskG'+dxStr)

	region_mask, xptsI, yptsI = cF.get_region_mask_pyproj(anc_data_pathT, proj, xypts_return=1)
	region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')



	folderPath=forcingPath+'/OIB/{}binned/{}/'.format(dxStr,yearT)
	days_list = os.listdir(folderPath)
	
	for file_day in days_list:

		print('File:', file_day)
		day_val = (pd.to_datetime(file_day[:8])-date_start).days

		try:
			# print(os.path.join(folderPath,file_day))
			snowDepthOIB=np.load(os.path.join(folderPath,file_day),allow_pickle=True)
			# transpose (when using old OIB files)
			snowDepthOIB = snowDepthOIB.T

		except:
			continue

		print('Num points in day:', np.size(snowDepthOIB))
		if (np.size(snowDepthOIB)==0):
			#nodata
			continue
		# if (day_val>259):
		# 	#beyond May 1st
		# 	continue

		# snow depth from NESOSIM output (budget)
		snowDepthM=cF.get_budgets2layers_day(['snowDepthTotalConc'], outPath, folderStr, day_val, totalOutStr)
		
		if grid_100:
			# snow here was loaded as 50x50 (M), grid to 100x100 (G) to compare with OIB
			snowDepthM = griddata((xptsM.flatten(), yptsM.flatten()), snowDepthM.flatten(), (xptsG, yptsG), method='linear')
			snowDepthM = ma.masked_invalid(snowDepthM)

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
		maskDay[np.where(snowDepthM<=0.04)]=1

		maskDay[np.where(snowDepthOIB>0.8)]=1
		maskDay[np.where(snowDepthM>0.8)]=1

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

	return xptsGMall, yptsGMall, snowOIBMall, snowDepthMMall

def calc_loglike(model, obs, uncert):
	# log likelihood for normal distribution
	# based on likelihood function exp (0.5*sum((model-obs)^2/uncert^2))
	return -0.5*np.sum((model - obs)**2/uncert**2)

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


def main(params):
	windPackFactorT, leadLossFactorT = params
	folderStr=precipVar+CSstr+'sf'+windVar+'winds'+driftVar+'drifts'+concVar+'sic'+'rho'+densityTypeT+'_IC'+str(IC)+'_DYN'+str(dynamicsInc)+'_WP'+str(windpackInc)+'_LL'+str(leadlossInc)+'_AL'+str(atmlossInc)+'_WPF'+str(windPackFactorT)+'_WPT'+str(windPackThreshT)+'_LLF'+str(leadLossFactorT)+'-'+dxStr+extraStr+outStr

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
		totalOutStr=''+folderStr+'-'+dateOut


		# get depth by year for given product
		_, _, snowDepthOIByr, snowDepthMMyr= getOIBNESOSIM(dx, folderStr, totalOutStr, year2, 'GSFC', reanalysis, grid_100=True)#, days_y, diff_y)
		snowDepthOIBAll.extend(snowDepthOIByr)
		snowDepthMMAll.extend(snowDepthMMyr)

	# calculate the log-likelihood
	log_p = calc_loglike(snowDepthMMAll, snowDepthOIBAll)

	return log_p
