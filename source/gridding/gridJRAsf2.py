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

import calendar
import pandas as pd
import xarray as xr

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

from config import reanalysis_raw_path
from config import forcing_save_path
from config import figure_path


def get_day_diff(day_wanted, day_start):
    '''day_wanted and day_start are of the form yyyy-mm-dd'''
    # first day is day 0?
    dw = pd.to_datetime(day_wanted)
    ds = pd.to_datetime(day_start)
    day_diff = dw - ds
    return(day_diff.days)


# loop through the days and grid

def calc_daily_means_jra_month(month, year):
	# calculate daily mean; returns a data array
	firstday,lastday = calendar.monthrange(year, int(month))
	prefix = '/data/kushner_group/JRA/srweq/'
	
	
#	fname = prefix + 'fcst_phy2m125.064_srweq.{}{}0100_{}{}{}21'.format(year,month,year,month,lastday)
	fname = prefix + 'fcst_phy2m125.064_srweq.{}010100_{}123121'.format(year,year)
	
	data = xr.open_dataset(fname,engine='cfgrib')
	data = data['srweqsfc'].loc['{}-{}'.format(year,month)]
#	sum_step = data['srweqsfc'].sum(axis=1)
	sum_step = data.sum(axis=1)

	daily_mean = sum_step.resample(time="1D").sum()/8
	return daily_mean



# constants

YEAR_START = 1980
YEAR_END = 2014
MONTH_START = 9
MONTH_END = 4
DAY_START = 1
DAY_END = 30

MONTHS_ALL = ['01','02','03','04','09','10','11','12']

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

varStr='sf'

OUT_PATH = '/users/jk/20/acabaj/jra-forcings/'

# loop over the years

first_iter = True

for year in range(YEAR_START, YEAR_END):
	year_path = OUT_PATH + str(year)
	if not os.path.exists(year_path):
		os.makedirs(year_path)
	for month in MONTHS_ALL:
		print('gridding for {}-{}'.format(month,year))
		# load data
		daily_mean = calc_daily_means_jra_month(month,year)
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

		# restrict lat/lon bounds
		daily_mean = daily_mean[:,:lowerLatidx,:]

		print(daily_mean.shape[0])
		# iterate over day:
		for i in range(daily_mean.shape[0]):
			interp = LinearNDInterpolator(tri,daily_mean[i].values.flatten())
			PrecipG = interp((xptsG,yptsG))
			# save the value for the day
			# calculate day string; day of year
			day_of_month = i+1
			day_of_year = get_day_diff('{}-{}-{}'.format(year,month,day_of_month),'{}-{}-{}'.format(year,'01','01'))
			# returns 0 for first day of year, as we want

			# save data
			PrecipG.dump(OUT_PATH+str(year)+'/JRA55'+varStr+dxStr+'-'+str(year)+'_d{:03d}v11_1'.format(day_of_year))

