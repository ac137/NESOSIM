""" gridERA5t2mdaily.py
	grid era5 temperatures
	
	Model written by Alek Petty (06/01/2020)
	Contact me for questions (alek.a.petty@nasa.gov)

	Input: ERA5 2m air temperatures
	Output: Gridded ERA5 2m temperatures

	Python dependencies:
		See below for the relevant module imports
		Also some function in utils.py

	Update history:
		05/01/2020: Version 1 (adapted from the earlier ERA-I script)
						- utilize xarray's resample and reduce functions to optimize the code.
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from glob import glob
from scipy.interpolate import griddata
import sys
sys.path.append('../')
import utils as cF
import os
import pyproj
import cartopy.crs as ccrs
import itertools
import pandas as pd
import xarray as xr

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

from config import reanalysis_raw_path
from config import forcing_save_path
from config import figure_path

#reanalysis_raw_path = '/data/kushner_group/ERA/t2m_uv_pr/'
#reanalysis_raw_path = '/data/kushner_group/acabaj/e5-daily/'
reanalysis_raw_path = '/users/jk/21/acabaj/e5_t2m_daily/'# for 2020
forcing_save_path = '/users/jk/21/acabaj/e5_t2m_for_nesosim/'
#forcing_save_path = '/users/jk/18/acabaj/NESOSIM/forcing_2020_23/'

print(forcing_save_path)
print(figure_path)
figure_path = '/users/jk/18/acabaj/NESOSIM/figures/'


def get_day_diff(day_wanted, day_start):
	'''day_wanted and day_start are of the form yyyy-mm-dd'''
	# first day is day 0?
	dw = pd.to_datetime(day_wanted)
	ds = pd.to_datetime(day_start)
	day_diff = dw - ds
	return(day_diff.days)

def get_ERA5_temps(data_pathT, yearT):
	
	# daily eg.  e5_t2m_uv_pr_daily_2018.nc
#	tempdata=xr.open_mfdataset(data_pathT+'/e5_t2m_uv_pr_daily_{}.nc'.format(yearT))
#	tempdata=xr.open_mfdataset(data_pathT+'/e5_t2m_daily_{}*.nc'.format(yearT))
	tempdata=xr.open_mfdataset(data_pathT+'/e5_t2m_daily_nh_{}*.nc'.format(yearT))
		
#	numDaysYearT=np.size(tempdata['time'][:])
	numDaysYearT=np.size(tempdata['valid_time'][:])
	print (numDaysYearT)

	lon = tempdata['longitude'][:]
	lowerlat=20
	tempdata=tempdata.where(tempdata.latitude>lowerlat, drop=True)
	
	lon = tempdata['longitude'][:]
	lat = tempdata['latitude'][:]
	tempdata_daily = tempdata['t2m']-273.15 #already daily for this case
#	xpts, ypts=proj(*np.meshgrid(lon, lat))
	return tempdata_daily
#	return xpts, ypts, tempdata_daily

#def get_ERA5_temps_hourly(proj, data_pathT, yearT):
	# hourly

#	tempdata=xr.open_mfdataset(data_pathT+'/e5_t2m_hourly_{}*.nc'.format(yearT))
#	numDaysYearT=np.size(tempdata['time'][:])/24
#	print(numDaysYearT)
	
#	lon = tempdata['longitude'][:]
#	lowerlat=20
		
#	tempdata=tempdata.where(tempdata.latitude>lowerlat, drop=True)
	
#	lon = tempdata['longitude'][:]
#	lat = tempdata['latitude'][:]
#	tempdata = tempdata.resample(time='1D').mean()
	
#	tempdata_daily = tempdata-273.15 #already daily for this case
	
#	xpts, ypts=proj(*np.meshgrid(lon, lat))
	# need to resample to daily

#	return xpts, ypts, tempdata_daily

YEAR_START = 2020
YEAR_END = 2021
MONTH_START = 1
MONTH_END = 12
DAY_START = 0
DAY_END = 30

MONTHS_ALL = ['01','02','03','04','09','10','11','12']
MONTHS_ALL = ['05','06','07','08']
#MONTHS_ALL = ['01']

LOWER_LAT = 30
dx=100000
ANC_DATA_PATH = '../../anc_data/'

xptsG, yptsG, latG, lonG, proj = cF.create_grid(dxRes=dx)
print(xptsG)
print(yptsG)

dxStr=str(int(dx/1000))+'km'
print(dxStr)


region_mask, xptsI, yptsI = cF.get_region_mask_pyproj(ANC_DATA_PATH, proj, xypts_return=1)
region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')

varStr='t2m'

first_iter = True
OUT_PATH = forcing_save_path


for year in range(YEAR_START, YEAR_END):
	year_path = OUT_PATH + str(year)
	if not os.path.exists(year_path):
		os.makedirs(year_path)
	daily_mean = get_ERA5_temps(reanalysis_raw_path,year)
	print(daily_mean)
	if first_iter:
	# do gridding with lon/lat (M for model (rean))
		latsM = daily_mean['latitude'].values
		lowerLatidx=int((90-LOWER_LAT)/(latsM[0]-latsM[1]))
		latsM=latsM[0:lowerLatidx]
		lonsM = daily_mean['longitude'].values
		# get points in projection
		xptsM, yptsM=proj(*np.meshgrid(lonsM, latsM))
		ptM_arr = np.array([xptsM.flatten(),yptsM.flatten()]).T
		# delaunay triangulation
		tri = Delaunay(ptM_arr)
		first_iter = False	
	for month in MONTHS_ALL:
		print('gridding for {}-{}'.format(month,year))
		# restrict lat/lon bounds
#		print(daily_mean)
		daily_mean = daily_mean[:,:lowerLatidx,:]
		daily_mean_sel = daily_mean.loc['{}-{}'.format(year,month)]

		print('month length')
		print(daily_mean_sel.shape[0])
		# iterate over day:
		for i in range(daily_mean_sel.shape[0]):
			interp = LinearNDInterpolator(tri,daily_mean_sel[i].values.flatten())
			t2mG = interp((xptsG,yptsG))
			# save the value for the day
			# calculate day string; day of year
			day_of_month = i+1
			day_of_year = get_day_diff('{}-{}-{}'.format(year,month,day_of_month),'{}-{}-{}'.format(year,'01','01'))
			# returns 0 for first day of year, as we want
			save_path = OUT_PATH+str(year)+'/ERA5'+varStr+dxStr+'-'+str(year)+'_d{:03d}v11'.format(day_of_year)
#			print(save_path)

			# save data
			t2mG.dump(OUT_PATH+str(year)+'/ERA5'+varStr+dxStr+'-'+str(year)+'_d{:03d}v11'.format(day_of_year))


#-- run main program
#if __name__ == '__main__':
#	for y in range(1991, 1991+1, 1):
#		print(y)
#		main(y)


	

