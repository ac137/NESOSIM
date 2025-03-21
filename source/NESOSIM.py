""" NESOSIM.py
	
	The NASA Euelerian Snow on Sea Ice Model (NESOSIM) v1.1 
	Model written by Alek Petty
	Contact me for questions (alek.a.petty@nasa.gov) or refer to the GitHub site (https://github.com/akpetty/NESOSIM)

	Run this python script with the run.py script in this same directory. 

	Input:
		Gridded/daily data of snowfall, ice drift, ice concentration, wind speeds

	Output:
		Gridded/daily data of the snow depth/density and snow budget terms.
		The DataOutput/MODELRUN/budgets/ netcdf files are all the snow budget terms needed for the analysis scripts/
		The DataOutput/MODELRUN/final/ netcdf files are the finalized netcdf files of the key variables, including metadata.

	Python dependencies:
		See below for the relevant module imports. Of note:
		xarray/pandas
		netCDF4
		matplotlib
		cartopy

		More information on installation is given in the README file.

	Update history:
		1st March 2018: Version 0.1
		1st October 2018: Version 1.0 (updated through review process)
		1st May 2020: Version 1.1 (updated for ICESat-2 processing, new domain using cartopy/pyproj, bug fixes)
		1st October 2020: Version 1.1 (bug fixes, replace masked array with nan throughout, smoothed dynamics terms)
		15th October 2020: Version 1.1 (new atmosphere-wind loss term, fix to transposed grids, clean up)

"""

import numpy as np
import numpy.ma as ma
import xarray as xr
import pandas as pd
import os
from glob import glob
from scipy.ndimage.filters import gaussian_filter

import utils as cF
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import datetime
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel


def calcLeadLoss(snowDepthT, windDayT, iceConcDaysT):
	""" Snow loss to leads due to winds

	Use a variable leadlossfactor parameter. This is relatively unconstrained!

	Args:
		snowDepthT (var): Daily gridded snowdepth 
		WindDayT (var): Daily gridded wind magnitude
		iceConcDaysT (var): Daily gridded ice concentration

	returns:
		snowLeadT (var): Snow lost from fresh snow layer

	Updates:
		v1.0 (during review process) added wind packing threshold

	"""

	windT= np.where(windDayT>windPackThresh, 1, 0)
	 
	snowLeadT = -(windT*leadLossFactor*deltaT*snowDepthT*windDayT*(1-iceConcDaysT)) #*iceConcDaysG[x]
	return snowLeadT

def calcAtmLoss(snowDepthT, windDayT):
	""" Snow lost to the atmosphere due to winds

	Using a multiple of the variable windlossfactor parameter. This is relatively unconstrained!

	Args:
		snowDepthT (var): Daily gridded snowdepth 
		WindDayT (var): Daily gridded wind magnitude
		
	returns:
		snowAtmLossT (var): Snow lost from fresh snow layer

	Updates:
		v1.1: new wind-atmosphere snow loss term
		v1.1 (May 2022): removed dependence on leadLossFactor and changed default value of atm_loss_factor (from 0.15 to 2e-8)

	"""

	windT= np.where(windDayT>windPackThresh, 1, 0)
	
	snowAtmLossT = -(windT*deltaT*snowDepthT*windDayT*atmLossFactor)
	return snowAtmLossT

def calcWindPacking(windDayT, snowDepthT0):
	""" Snow pack densification through wind packing

	Calculated using the amount of snow packed down and the 
	difference in density between the fresh snow density and the old snow density

	Args:
		snowDepthT (var): Daily gridded snowdepth 
		WindDayT (var): Daily gridded wind magnitude
		iceConcDaysT (var): Daily gridded ice concentration

	returns:
		snowWindPackLossT (var): Snow lost from fresh snow layer
		snowWindPackGainT (var): Snow gained to old snow layer
		snowWindPackNetT (var): Net snow gain/loss

	"""


	windT= np.where(windDayT>windPackThresh, 1, 0)
	
	# snow loss from fresh layer through wind packing to old layer
	snowWindPackLossT=-windPackFactor*deltaT*windT*snowDepthT0 #*iceConcDaysG[x]

	# snow gain to old layer through wind packing from fresh layer
	snowWindPackGainT=windPackFactor*deltaT*windT*snowDepthT0*(snowDensityFresh/snowDensityOld) #*iceConcDaysG[x]
	
	snowWindPackNetT=snowWindPackLossT+snowWindPackGainT#*iceConcDaysG[x]
	return snowWindPackLossT, snowWindPackGainT, snowWindPackNetT

def fillMaskAndNaNWithZero(arr):
	""" Helper function: Fill masked and nan values in an array 
	with 0, in place

	Args:
		arr (var): A numpy ndarray
	returns:
		None (performs operation in place)

	"""
	arr[np.isnan(arr)] = 0.
	arr = ma.filled(arr, 0.)
	arr[~np.isfinite(arr)] = 0.

def fill_nan_no_negative(arr, region_maskG, negative_to_zero=True):
	""" Helper function: Fill masked with nan and don't allow negative snow depths if selected

	Args:
		arr (var): A numpy ndarray x, y
	returns:
		None (performs operation in place)

	"""

	

	# Set infinte snow depths to nan
	arr[~np.isfinite(arr)]=np.nan
	arr[~np.isfinite(arr)]=np.nan
	
	# Set snow depths over land/coasts to nan (changed from zero)
	arr[np.where(region_maskG>10)]=np.nan
	arr[np.where(region_maskG>10)]=np.nan

	# Set snow depths over lakes (lake sic included in CDR) to nan (changed from zero)
	arr[np.where(region_maskG<1)]=np.nan

	# Set negative snow to zero.
	if (negative_to_zero):
		arr[np.where(arr<0.)]=0.
	#arr[x+1][np.where(np.isnan(arr[x+1]))]=0.


def smooth_snow(arr, stddev_val=1,x_size_val=3, y_size_val=3):
	""" Smooths array using a Gaussian kernal (from astropy)

	Args:
		arr (var): A numpy ndarray x, y
		stddev_val (constant): gaussian width of filter
		x_size_val (constant): grid-cell width in x
		y_size_val (constant): grid-cell width in y
		
	returns:
		None (performs operation in place)
	"""
	
	# x_stddev applies to both x and y if y isn't specified.
	kernel = Gaussian2DKernel(x_stddev=stddev_val, x_size=x_size_val, y_size=x_size_val)
	arr = convolve(arr, kernel)
	
	return arr

def calcDynamics(driftGday, snowDepthsT, dx):
	""" Snow loss/gain from ice dynamics

	Args:
		driftGday (var): Daily gridded ice drift
		snowDepthT (var): Daily gridded snowdepth 
		dx (var): grid spacing

	returns:
		snowAdvAllT (var): Snow change through advection (gain is positive)
		snowDivAllT (var): Snow change through convergence/divergence (convergence is positive)
		
	"""

	#------------  Snow change from ice divergence/convergence. Convergence is positive
	dhsvelxdxDiv = snowDepthsT*np.gradient(driftGday[0]*deltaT, dx, axis=(1)) #convert from m/s to m per day, #1 here is the columns, so in the x direction
	dhsvelydyDiv = snowDepthsT*np.gradient(driftGday[1]*deltaT, dx, axis=(0)) #0 here is the rows, so in the y direction
	# Snow divergence
	snowDivAllT= -(dhsvelxdxDiv + dhsvelydyDiv)

	#------------ Snow change from ice advection. Advetion away from a grid-cell is negative
	dhsvelxdxAdv = driftGday[0]*deltaT*np.gradient(snowDepthsT, dx, axis=(2)) #convert from m/s to m per day, #1 here is the columns, so in the x direction
	dhsvelydyAdv = driftGday[1]*deltaT*np.gradient(snowDepthsT, dx, axis=(1))  #0 here is the rows, so in the y direction
	# Snow advection
	snowAdvAllT= -(dhsvelxdxAdv + dhsvelydyAdv)

	# Set bad values to zero
	fillMaskAndNaNWithZero(snowAdvAllT[0])
	fillMaskAndNaNWithZero(snowAdvAllT[1])
	fillMaskAndNaNWithZero(snowDivAllT[0])
	fillMaskAndNaNWithZero(snowDivAllT[1])
	

	return snowAdvAllT, snowDivAllT
	

def calcMelt(t2m_day, method='linear',density_weight=True):
	''' np.ndarray, str -> np.ndarray, np.ndarray
	for a given day, given gridded temperature t2m_day (in celsius),
	calculate and return a arrays of snow melt (in m) for the snow budget,
	for the upper (0) and lower (1) layers
	makes use of global variables: 
	- meltThresh determines the minimum temperature for melt to occur.
	- consecutive_melt_day_count for some methods to check how long temperature
	has been above melt threshold
	
	- meltFactor is a scaling factor
	method (str) determines which melt method is applied;
		- 'constant': melt = meltFactor when t2m_day >= meltThresh
		- 'linear': melt = meltFactor*t2m_day when t2m_day >= meltThresh
		- 'melt_day_constant': as with 'constant' but temperature needs
		to be above meltThresh for a sufficient number of days
		- 'melt_day_linear' as with 'linear' but temperature needs to be
		aboce meltThresh for a sufficient number of days
	'''
	
	# I think python version is too old for match/case statements 
	if method=='constant':
	# constant melt above threshold option
		melting_grid_points = (t2m_day >= meltThresh)*meltFactor 
	elif method=='linear':
	# linear dependence on temperature when above threshold
		melting_grid_points = (t2m_day >= meltThresh)*meltFactor*t2m_day
	elif method=='melt_day_constant':
		# constant melt after specific number of days above melt threshold
		# need a threshold for that also but just going for now

		# first update the melt count
		# consecutive_melt_day_count is a global var; updated in place
		# maybe not best to use global var from this and pass it between 
		# functions instead but can think about architecture later
		updateMeltDayCount(consecutive_melt_day_count, t2m_day)

		# now check if melt day count is greater than count threshold
		count_threshold = 3 # setting to arbitrary number for now
		melting_grid_points = (consecutive_melt_day_count > count_threshold)*meltFactor
	elif method=='melt_day_linear':
		# linear melt (function of temperature) after specific number of days 
		# above melt threshold
		updateMeltDayCount(consecutive_melt_day_count, t2m_day)

		# now check if melt day count is greater than count threshold
		count_threshold = 3 # setting to arbitrary number for now
		melting_grid_points = (consecutive_melt_day_count > count_threshold)*meltFactor*t2m_day


	# weighting melt by layer density
	if density_weight:
		melt_upper_layer = melting_grid_points/snowDensityFresh
		melt_lower_layer = melting_grid_points/snowDensityOld
	else:
	# same melt in each layer 
	# (maybe adjust to weigh by average density because otherwise 
	# factors will compare strangely here)
		melt_upper_layer = melting_grid_points
		melt_lower_layer = melting_grid_points

	# don't need to check if snow depth is > 0 here because there's already a function to fix negative values if those happen
	return melt_upper_layer, melt_lower_layer
	

def calcBudget(xptsG, yptsG, snowDepths, iceConcDayT, precipDayT, driftGdayT, windDayT, tempDayT, 
	density, precipDays, iceConcDays, windDays, tempDays, snowAcc, snowOcean, snowAdv, 
	snowDiv, snowLead, snowAtm, snowWindPackLoss, snowWindPackGain, snowWindPack, region_maskG, dx, x, dayT,
	densityType='variable', dynamicsInc=1, leadlossInc=1, windpackInc=1, atmlossInc=0,meltlossInc=0):
	""" Snow budget calculations

	Args:
		xptsG (var, x/y): x coordinates of grid
		yptsG (var, x/y): y coordinates of grid
		snowDepths (var, day/x/y): daily snow depth grid
		iceConcDayT (var, x/y): ice concentration for that day
		precipDayT (var, x/y): precip for that day
		driftGdayT (var, x/y): ice drift for that day
		windDayT (var, x/y): wind speed magnitude for that day
		tempDayT (var, x/y): near surface air temperture for that day
		density (var, x/y/j): 2 layer snow density

	returns:
		Updated snow budget arrays
		
	"""

	precipDays[x]=precipDayT
	iceConcDays[x]=iceConcDayT
	windDays[x]=windDayT
	tempDays[x]=tempDayT

	if (densityType=='clim'):
		# returns a fixed density value assigned to all grid cells based on climatology. 
		# Applies the same value to both snow layers.
		snowDensityNew=cF.densityClim(dayT, ancDataPath)
	else:
		# Two layers so a new snow density and an evolving old snow density
		snowDensityNew=snowDensityFresh
		
	# Convert precip to m/day
	precipDayDelta=precipDayT/snowDensityNew

	# ------ Snow accumulation
	snowAccDelta= (precipDayDelta * iceConcDayT)
	snowAcc[x+1] = snowAcc[x] + snowAccDelta

	# ------ Ocean freshwater flux
	snowOceanDelta= -(precipDayDelta * (1-iceConcDayT))
	snowOcean[x+1] = snowOcean[x] + snowOceanDelta

	# ------ Snow/ice dynamics calulation

	if (dynamicsInc==1):
		snowAdvDelta, snowDivDelta = calcDynamics(driftGdayT, snowDepths[x], dx)

		# Smooth dynamics terms as generally quite noisy
		snowAdvDelta[0]=smooth_snow(snowAdvDelta[0])
		snowAdvDelta[1]=smooth_snow(snowAdvDelta[1])
		snowDivDelta[0]=smooth_snow(snowDivDelta[0])
		snowDivDelta[1]=smooth_snow(snowDivDelta[1])
		
		fill_nan_no_negative(snowAdvDelta[0], region_maskG, negative_to_zero=False)
		fill_nan_no_negative(snowAdvDelta[1], region_maskG, negative_to_zero=False)
		fill_nan_no_negative(snowDivDelta[0], region_maskG, negative_to_zero=False)
		fill_nan_no_negative(snowDivDelta[1], region_maskG, negative_to_zero=False)

	else:
		snowAdvDelta=np.zeros((iceConcDayT.shape))
		snowDivDelta=np.zeros((iceConcDayT.shape))
	
	snowAdv[x+1] = snowAdv[x] + snowAdvDelta[0]+ snowAdvDelta[1]
	snowDiv[x+1] = snowDiv[x] + snowDivDelta[0]+ snowDivDelta[1]

	# ------ Lead loss calulation

	if (leadlossInc==1):
		snowLeadDelta= calcLeadLoss(snowDepths[x, 0], windDayT, iceConcDayT)
	else:
		snowLeadDelta=np.zeros((iceConcDayT.shape))

	snowLead[x+1]=snowLead[x] + snowLeadDelta

	# ------ Wind loss calulation

	if (atmlossInc==1):
		snowAtmDelta= calcAtmLoss(snowDepths[x, 0], windDayT)
	else:
		snowAtmDelta=np.zeros((iceConcDayT.shape))

	snowAtm[x+1]=snowAtm[x] + snowAtmDelta

	#---------- Wind packing calulation

	if (windpackInc==1):
		snowWindPackLossDelta, snowWindPackGainDelta, snowWindPackNetDelta=calcWindPacking(windDayT, snowDepths[x, 0])
	else:
		snowWindPackLossDelta =np.zeros((iceConcDayT.shape))
		snowWindPackGainDelta=np.zeros((iceConcDayT.shape))
		snowWindPackNetDelta=np.zeros((iceConcDayT.shape))

	snowWindPackLoss[x+1]=snowWindPackLoss[x]+snowWindPackLossDelta
	snowWindPackGain[x+1]=snowWindPackGain[x]+snowWindPackGainDelta
	snowWindPack[x+1]=snowWindPack[x]+snowWindPackNetDelta

	#------------ Update snow depths
	
	if meltlossInc==1:
		# calc melt now returns a tuple of (upper layer, lower layer) melt
		snowMeltLossDelta = calcMelt(tempDayT)
	else:
		# no melt
		snowMeltLossDelta = (0,0)

	# New (upper) layer
	snowDepths[x+1, 0]=snowDepths[x, 0]+snowAccDelta  +snowWindPackLossDelta + snowLeadDelta + snowAtmDelta +snowAdvDelta[0]+snowDivDelta[0] +snowMeltLossDelta[0]#+snowRidgeT
	# Old snow layer
	snowDepths[x+1, 1]=snowDepths[x, 1] +snowWindPackGainDelta + snowAdvDelta[1] + snowDivDelta[1]+snowMeltLossDelta[1] #+ snowDcationT

	# Fill negatives and set nans
	fill_nan_no_negative(snowDepths[x+1, 0], region_maskG)
	fill_nan_no_negative(snowDepths[x+1, 1], region_maskG)
	
	if (densityType=='clim'):
		# returns a fixed density value assigned to all grid cells based on climatology. 
		# Applies the same value to both snow layers.
		
		density[x+1]=snowDensityNew
		# mask over land, lakes and coast
		density[x+1][np.where(region_maskG>10)]=np.nan
		density[x+1][np.where(region_maskG<1)]=np.nan
		density[x+1][np.where(iceConcDayT<minConc)]=np.nan
		density[x+1][np.where((snowDepths[x+1][0]+snowDepths[x+1][1])<minSnowD)]=np.nan
	else:
		# Two layers so a new snow density and an evolving old snow density	
		density[x+1]=densityCalc(snowDepths[x+1], iceConcDayT, region_maskG)


	

def genEmptyArrays(numDaysT, nxT, nyT):
	""" 
	Declare empty arrays to store the various budget terms

	"""
	
	precipDays=np.zeros((numDaysT, nxT, nyT)) 
	iceConcDays=np.zeros((numDaysT, nxT, nyT)) 
	windDays=np.zeros((numDaysT, nxT, nyT)) 
	tempDays=np.zeros((numDaysT, nxT, nyT)) 

	snowDepths=np.zeros((numDaysT, 2, nxT, nyT))
	density=np.zeros((numDaysT, nxT, nyT))

	snowDiv=np.zeros((numDaysT, nxT, nyT))
	snowAdv=np.zeros((numDaysT, nxT, nyT))
	snowAcc=np.zeros((numDaysT, nxT, nyT))
	snowOcean=np.zeros((numDaysT, nxT, nyT))
	snowWindPack=np.zeros((numDaysT, nxT, nyT))
	snowWindPackLoss=np.zeros((numDaysT, nxT, nyT))
	snowWindPackGain=np.zeros((numDaysT, nxT, nyT))
	snowLead=np.zeros((numDaysT, nxT, nyT))
	snowAtm=np.zeros((numDaysT, nxT, nyT))
	

	return precipDays, iceConcDays, windDays, tempDays, snowDepths, density, snowDiv, snowAdv, snowAcc, snowOcean, snowWindPack, \
	snowWindPackLoss, snowWindPackGain, snowLead, snowAtm


def loadData(yearT, dayT, precipVar, windVar, concVar, driftVar, dxStr, extraStr):
	""" Load daily forcings

	Temp added transpose to convert grid to proper row/column index. 

	"""
	dayStr='%03d' %dayT
	
	#------- Read in precipitation -----------
	try:
		# precip_path = forcingPath+'Precip/'+precipVar+'/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr
		precip_path = os.path.join(forcingPath,'Precip',precipVar,str(yearT),precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		print('Loading gridded snowfall forcing from:', precip_path)
		precipDayG=np.load(precip_path, allow_pickle=True)
		
	except:
		if (dayStr=='365'):
			precip_path = os.path.join(forcingPath,'Precip',precipVar,str(yearT),precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+'364'+extraStr)
			precipDayG = np.load(precip_path)
			# precipDayG=np.load(forcingPath+'Precip/'+precipVar+'/sf/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+'364'+extraStr, allow_pickle=True)
			print('no leap year data, used data from the previous day')
		else:
			print('No precip data so exiting!')
			exit()
	
	#------- Read in wind magnitude -----------
	try:
		# wind_path = forcingPath+'Winds/'+windVar+'/'+str(yearT)+'/'+windVar+'winds'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr
		wind_path = os.path.join(forcingPath,'Winds',windVar,str(yearT),windVar+'winds'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		print('Loading gridded wind forcing from:', wind_path)
		windDayG=np.load(wind_path, allow_pickle=True)
		
	except:
		if (dayStr=='365'):
			print('no leap year data, using data from the previous day')
			wind_path = os.path.join(forcingPath,'Winds',windVar,str(yearT),windVar+'winds'+dxStr+'-'+str(yearT)+'_d'+364+extraStr)
			windDayG=np.load(wind_path, allow_pickle=True)
		
		else:
			print('No wind data so exiting!')
			exit()

	#------- Read in ice concentration -----------
	try:
		# ice_path = forcingPath+'IceConc/'+concVar+'/'+str(yearT)+'/iceConcG_'+concVar+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr
		ice_path = os.path.join(forcingPath,'IceConc',concVar,str(yearT),'iceConcG_'+concVar+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		print('Loading gridded ice conc forcing from:', ice_path)
		iceConcDayG=np.load(ice_path, allow_pickle=True)
		
	except:
		if (dayStr=='365'):
			ice_path = os.path.join(forcingPath,'IceConc',concVar,str(yearT),'iceConcG_'+concVar+dxStr+'-'+str(yearT)+'_d'+364+extraStr)
			print('no leap year data, using data from the previous day')
			iceConcDayG=np.load(ice_path, allow_pickle=True)
	
		else:
			print('No ice conc data so exiting!')
			exit()

	# fill with zero
	iceConcDayG[~np.isfinite(iceConcDayG)]=0.
	
	#------- Read in ice drifts -----------
	try:
		# drift_path = forcingPath+'IceDrift/'+driftVar+'/'+str(yearT)+'/'+driftVar+'_driftG'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr
		drift_path = os.path.join(forcingPath,'IceDrift',driftVar,str(yearT),driftVar+'_driftG'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		print('Loading gridded ice drift forcing from:', drift_path)
		driftGdayG=np.load(drift_path, allow_pickle=True)	

	except:
		# if no drifts exist for that day then just set drifts to nan array (i.e. no drift).
		print('No drift data')
		driftGdayG = np.empty((2, iceConcDayG.shape[0], iceConcDayG.shape[1]))
		driftGdayG[:] = np.nan

	driftGdayG = ma.filled(driftGdayG, np.nan)
	#print(driftGdayG)

	#------- Read in temps (not currently used, placeholder) -----------
	temp_path = forcingPath+'/Temp/'+precipVar+'/t2m/'+str(yearT)+'/{}t2m'.format(precipVar)+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr
	print(temp_path)
	try:
		print('Loading gridded temperature data')
		tempDayG=np.load(forcingPath+'/Temp/'+precipVar+'/t2m/'+str(yearT)+'/{}t2m'.format(precipVar)+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr, allow_pickle=True)
		print('mean of gridded temperature',np.nanmean(tempDayG))
	except:
		# if no temperatures exist just set to nan
		print('No temp data')
		tempDayG = np.empty((iceConcDayG.shape[0], iceConcDayG.shape[1]))
		tempDayG[:] = np.nan
		#tempDayG=ma.masked_all((iceConcDayG.shape[0], iceConcDayG.shape[1]))
	
	return iceConcDayG, precipDayG, driftGdayG, windDayG, tempDayG

def densityCalc(snowDepthsT, iceConcDayT, region_maskT):
	"""Assign initial density based on snow depths

	dropped ice conc mask as based on old time step
	"""		

	densityT=((snowDepthsT[0]*snowDensityFresh) + (snowDepthsT[1]*snowDensityOld))/(snowDepthsT[0]+snowDepthsT[1]) #+ densDcationT
	
	densityT[np.where(densityT>snowDensityOld)]=snowDensityOld
	densityT[np.where(densityT<snowDensityFresh)]=snowDensityFresh

	densityT[np.where(region_maskT<1)]=np.nan
	densityT[np.where(region_maskT>10)]=np.nan
	densityT[np.where((snowDepthsT[0]+snowDepthsT[1])<minSnowD)]=np.nan

	return densityT

def doyToMonth(day, year):
	""" given the day-of-year day and the year, return an integer
	corresponding to the month during which the day occurs
	"""
	date_fmt = np.datetime64('{}-01-01'.format(year)) + np.timedelta64(day-1,'D')
	return date_fmt.astype(object).month

def applyScaling(product,factor,scaling_type='mul'):
	"""Apply a scaling factor to a given product; the factor
	must either be a scalar or have the same dimensions as
	the product
	
	"""
	if scaling_type=='mul':
		# multiplicative scaling
		product_scaled = product*factor

	return product_scaled


def updateMeltDayCount(melt_count_array, tempday):
	# towards doing a melt day count process?
	# use melt threshold meltThresh
	# see if this is a melt day or not and update the consecutive melt
	# day count array

	melt_occurring = tempday >= meltThresh

	# where melt is not occurring, reset to 0
	melt_count_array[~melt_occurring] = 0
	melt_count_array[melt_occurring] += 1

	# return melt_count_array # return statement not necessarily needed



def main(year1, month1, day1, year2, month2, day2, outPathT='.', forcingPathT='.', anc_data_pathT='../anc_data/', figPathT='../Figures/', 
	precipVar='ERA5', windVar='ERA5', driftVar='OSISAF', concVar='CDR', icVar='ERAI', densityTypeT='variable', 
	outStr='', extraStr='', IC=2, windPackFactorT=0.1, windPackThreshT=5., leadLossFactorT=0.1, atmLossFactorT=2.2e-8, meltThreshT=1,meltFactorT=-0.001,dynamicsInc=1, leadlossInc=1, 
	windpackInc=1, atmlossInc=0, saveData=1, plotBudgets=1, plotdaily=1, meltlossInc=0, saveFolder='', dx=50000,scaleCS=False):
	""" 

	Main model function

	Args:
		The various model configuration parameters

	"""

	#------- Create map projection
	xptsG, yptsG, latG, lonG, proj = cF.create_grid(dxRes=dx)
	nx=xptsG.shape[0]
	ny=xptsG.shape[1]

	dxStr=str(int(dx/1000))+'km'
	print(nx, ny, dxStr)

	# Assign some global parameters
	global dataPath, forcingPath, outPath, ancDataPath
	
	# outPath=outPathT+dxStr+'/'
	# forcingPath=forcingPathT+dxStr+'/'
	outPath=os.path.join(outPathT,dxStr)
	forcingPath=os.path.join(forcingPathT,dxStr)
	ancDataPath=anc_data_pathT
	print('OutPath:', outPath)
	print('forcingPath:', forcingPath)
	print('ancDataPath:', ancDataPath)

	# Assign density of the two snow layers
	global snowDensityFresh, snowDensityOld, minSnowD, minConc, leadLossFactor, atmLossFactor, windPackThresh, windPackFactor, deltaT, meltThresh, meltFactor
	snowDensityFresh=200. # density of fresh snow layer
	snowDensityOld=350. # density of old snow layer
	minSnowD=0.02 # minimum snow depth for a density estimate
	minConc=0.15 # mask budget values with a concentration below this value

	deltaT=60.*60.*24. # time interval (seconds in a day)

	region_mask, xptsI, yptsI, _, _ = cF.get_region_mask_pyproj(anc_data_pathT, proj, xypts_return=1)
	region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')

	leadLossFactor=leadLossFactorT # Snow loss to leads coefficient
	windPackThresh=windPackThreshT # Minimum winds needed for wind packing
	windPackFactor=windPackFactorT # Fraction of snow packed into old snow layer
	atmLossFactor=atmLossFactorT # Snow loss to atmosphere coefficient
	meltThresh = meltThreshT # threshold at or above which melt occurs
	meltFactor = meltFactorT # factor controlling melt rate

	#---------- Current year
	yearCurrent=year1
	
	#--------- Get time period info
	startDay, numDays, numDaysYear1, dateOut= cF.getDays(year1, month1, day1, year2, month2, day2)
	print (startDay, numDays, numDaysYear1, dateOut)

	# make this into a small function
	dates=[]
	for x in range(0, numDays):
		#print x
		date = datetime.datetime(year1, month1+1, day1+1) + datetime.timedelta(x)
		#print (int(date.strftime('%Y%m%d')))
		dates.append(int(date.strftime('%Y%m%d')))
	#print(dates)
	
	CSstr = ''
	if scaleCS:
		# load scaling factors; assumes scaling factors are in same directory as NESOSIM.py
		monthlyScalingFactors = xr.open_dataset('{}scale_coeffs_{}_{}_v2.nc'.format(ancDataPath, precipVar, dxStr))['scale_factors']
		CSstr = 'CSscaled'

	#------ create output strings and file paths -----------
	saveStr= precipVar+CSstr+'sf'+windVar+'winds'+driftVar+'drifts'+concVar+'sic'+'rho'+densityTypeT+'_IC'+str(IC)+'_DYN'+str(dynamicsInc)+'_WP'+str(windpackInc)+'_LL'+str(leadlossInc)+'_AL'+str(atmlossInc)+'_WPF'+str(windPackFactorT)+'_WPT'+str(windPackThreshT)+'_LLF'+str(leadLossFactorT)+'-'+dxStr+extraStr+outStr+'-'+dateOut
	saveStrNoDate=precipVar+CSstr+'sf'+windVar+'winds'+driftVar+'drifts'+concVar+'sic'+'rho'+densityTypeT+'_IC'+str(IC)+'_DYN'+str(dynamicsInc)+'_WP'+str(windpackInc)+'_LL'+str(leadlossInc)+'_AL'+str(atmlossInc)+'_WPF'+str(windPackFactorT)+'_WPT'+str(windPackThreshT)+'_LLF'+str(leadLossFactorT)+'-'+dxStr+extraStr+outStr
	
	print ('Saving to:', saveStr)
	 #'../../DataOutput/'

	# savePath=outPath+saveFolder+'/'+saveStrNoDate
	savePath = os.path.join(outPath,saveFolder,saveStrNoDate)
	# Declare empty arrays for compiling budgets
	if not os.path.exists(os.path.join(savePath,'budgets')):
		os.makedirs(os.path.join(savePath,'budgets'))
	if not os.path.exists(os.path.join(savePath,'final')):
		os.makedirs(os.path.join(savePath,'final'))

	global figpath
    # figpath = os.path.join(figpathT,'Diagnostic',dxStr,saveStrNoDate)
	figpath=figPathT+'/Diagnostic/'+dxStr+'/'+saveStrNoDate+'/'
	if not os.path.exists(figpath):
		os.makedirs(figpath)
	if not os.path.exists(figpath+'/daily_snow_depths/'):
		os.makedirs(figpath+'/daily_snow_depths/')

	precipDays, iceConcDays, windDays, tempDays, snowDepths, density, snowDiv, snowAdv, snowAcc, snowOcean, snowWindPack, snowWindPackLoss, snowWindPackGain, snowLead, snowAtm = genEmptyArrays(numDays, nx, ny)

	print('IC:', IC)
	if (IC>0):
		if (IC==1):
			# August Warren climatology snow depths
			ICSnowDepth = np.load(forcingPath+'InitialConditions/AugSnow'+dxStr, allow_pickle=True)
			print('Initialize with August Warren climatology')
		
		elif (IC==2):
			# Petty initiail conditions
			ic_path = os.path.join(forcingPath,'InitialConditions',icVar,'ICsnow'+dxStr+'-'+str(year1)+extraStr)
			# print(forcingPath+'InitialConditions/'+icVar+'/ICsnow'+dxStr+'-'+str(year1)+extraStr)
			try:
				ICSnowDepth = np.load(ic_path, allow_pickle=True)
				print('Initialize with new v1.1 scaled initial conditions')
				print(np.amax(ICSnowDepth))
			except:
				print('No initial conditions file available')

		iceConcDayG, precipDayG, driftGdayG, windDayG, tempDayG =loadData(year1, startDay, precipVar, windVar, concVar, driftVar, dxStr, extraStr)
		ICSnowDepth[np.where(iceConcDayG<minConc)]=0

		#--------Split the initial snow depth over both layers
		snowDepths[0, 0]=ICSnowDepth*0.5
		snowDepths[0, 1]=ICSnowDepth*0.5

	#pF.plotSnow(m, xptsG, yptsG, densityT, date_string=str(startDay-1), out=figpath+'/Snow/2layer/densityD'+driftP+extraStr+reanalysisP+varStr+'_sy'+str(year1)+'d'+str(startDay)+outStr+'T0', units_lab=r'kg/m3', minval=180, maxval=360, base_mask=0, norm=0, cmap_1=cm.viridis)


	# create array to store melt day count. maybe later should have genemptyarrays handle this but just for now
	# make this a global variable?
	global consecutive_melt_day_count
	consecutive_melt_day_count = np.zeros(precipDays[0].shape)

	# Loop over days 
	for x in range(numDays-1):	
		day = x+startDay
		
		if (day>=numDaysYear1):
			# If day goes beyond the number of days in initial year, jump to the next year
			day=day-numDaysYear1
			yearCurrent=year2
		
		print ('Day of year:', day)
		print ('Date:', dates[x])
		
		#-------- Load daily data 
		iceConcDayG, precipDayG, driftGdayG, windDayG, tempDayG =loadData(yearCurrent, day, precipVar, windVar, concVar, driftVar, dxStr, extraStr)
		print('temperature data', np.mean(tempDayG))
		
		#-------- Apply CloudSat scaling if used
		if scaleCS:
			currentMonth = doyToMonth(day, yearCurrent) # get current month
			scalingFactor = monthlyScalingFactors.loc[currentMonth,:,:] # get scaling factor for current month
			# apply scaling to current day's precipitation
			precipDayG = applyScaling(precipDayG, scalingFactor,scaling_type='mul').values



		#-------- Calculate snow budgets
		calcBudget(xptsG, yptsG, snowDepths, iceConcDayG, precipDayG, driftGdayG, windDayG, tempDayG,
			density, precipDays, iceConcDays, windDays, tempDays, snowAcc, snowOcean, snowAdv, 
			snowDiv, snowLead, snowAtm, snowWindPackLoss, snowWindPackGain, snowWindPack, region_maskG, dx, x, day,
			densityType=densityTypeT, dynamicsInc=dynamicsInc, leadlossInc=leadlossInc, windpackInc=windpackInc, atmlossInc=atmlossInc, meltlossInc=meltlossInc)
		
		if (plotdaily==1):
			cF.plot_gridded_cartopy(lonG, latG, snowDepths[x+1, 0]+snowDepths[x+1, 1], proj=ccrs.NorthPolarStereo(central_longitude=-45), date_string='', out=figpath+'daily_snow_depths/snowTot_'+saveStrNoDate+str(x), units_lab='m', varStr='Snow depth', minval=0., maxval=0.6)
	
	#------ Load last data 
	iceConcDayG, precipDayG, _, windDayG, tempDayG =loadData(yearCurrent, day+1, precipVar, windVar, concVar, driftVar, dxStr, extraStr)
	precipDays[x+1]=precipDayG
	iceConcDays[x+1]=iceConcDayG
	windDays[x+1]=windDayG
	tempDays[x+1]=tempDayG
	print(saveStr)
	
	
	if (saveData==1):
		# Output snow budget terms to netcdf datafiles
		cF.OutputSnowModelRaw(savePath, 'NESOSIMv11_budget_'+dateOut, snowDepths, density, precipDays, iceConcDays, windDays, snowAcc, snowOcean, snowAdv, snowDiv, snowLead, snowAtm, snowWindPack)
		cF.OutputSnowModelFinal(savePath, 'NESOSIMv11_'+dateOut, lonG, latG, xptsG, yptsG, snowDepths[:, 0]+snowDepths[:, 1], (snowDepths[:, 0]+snowDepths[:, 1])/iceConcDays, density, iceConcDays, precipDays, windDays, tempDays, dates)

	if (plotBudgets==1):
		# Plot final snow budget terms 
		cF.plot_budgets_cartopy(lonG, latG, precipDayG, windDayG, snowDepths[x+1], snowOcean[x+1], snowAcc[x+1], snowDiv[x+1], \
		snowAdv[x+1], snowLead[x+1], snowAtm[x+1], snowWindPack[x+1], snowWindPackLoss[x+1], snowWindPackGain[x+1], density[x+1], dates[-1], figpath, totalOutStr='budgetplot')

