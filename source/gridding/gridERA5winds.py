
""" gridERA5sf.py
	
	Script to grid the ERA5 wind data
	Model code written by Alek Petty (05/01/2020)
	Contact me for questions (alek.a.petty@nasa.gov)

	Input: Hourly gridded ERA5 snowfall data (ERA5 grid)
	Output: Gridded daily ERA5 snowfall data (NESOSIM grid)

	Python dependencies:
		See below for the relevant module imports
		Also reads in some functions in utils.py

	Update history:
		12/18/2018: Version 1
		05/01/2020: Version 2: Changed from using Basemap to pyproj for projection transformation (e.g. https://github.com/pyproj4/pyproj/blob/master/docs/examples.rst)
							Changed from a 100 km to 50 km grid.
							Changed from some Basemap Polar Stereographic grid to the official NSIDC grid ("epsg:3413") https://epsg.io/3413
		03/12/2021: Version 3: introduce weights to speed up gridding

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
import xarray as xr

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

#from config import reanalysis_raw_path, forcing_save_path, figure_path

forcing_save_path = '/mnt/ccrp/data1/cabaja/snow_modelling/nesosim_gridded_data/'
reanalysis_raw_path = '/mnt/ccrp/data1/cabaja/reanalysis_data/ERA5/uv_hourly_nh/'
figure_path = '/mnt/ccrp/data1/cabaja/snow_modelling/NESOSIM/figures/'


def main(year, startMonth=0, endMonth=11, dx=100000, extraStr='v11', data_path=reanalysis_raw_path+'ERA5/', out_path=forcing_save_path+'Winds/ERA5/', fig_path=figure_path+'Winds/ERA5/', anc_data_path='../../anc_data/'):

	xptsG, yptsG, latG, lonG, proj = cF.create_grid(dxRes=dx)
	print(xptsG)
	print(yptsG)

	dxStr=str(int(dx/1000))+'km'
	print(dxStr)


	region_mask, xptsI, yptsI, _, _ = cF.get_region_mask_pyproj(anc_data_path, proj, xypts_return=1)
	region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')

	varStr='WindMag'

	if not os.path.exists(fig_path):
		os.makedirs(fig_path)

	yearT=year

	numDays=cF.getLeapYr(year)
	if (numDays>365):
		monIndex = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
	else:
		monIndex = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

	if not os.path.exists(out_path+'/'+str(year)):
		os.makedirs(out_path+'/'+str(year))

	startDay=monIndex[startMonth]

	if (endMonth>11):
		endDay=monIndex[endMonth+1-12]+monIndex[-1]-1
	else:
		endDay=monIndex[endMonth+1]

	calc_weights = 1 # start as one to calculate weightings then gets set as zero for future files


    # reworking because this is being finicky. just load the data all at once initially and then calculate
    # this only works if doing one month at a time so try that I guess
	dayStr0='%03d' %startDay
	month0=np.where(startDay-np.array(monIndex)>=0)[0][-1]
	monStr0='%02d' %(month0+1)

	print('loading data')
	f1 = xr.open_dataset('/mnt/ccrp/data1/cabaja/reanalysis_data/ERA5/uv_hourly_nh/e5_u_v_hourly_nh_{}_{}.nc'.format(yearT,monStr0),chunks={'valid_time':24})
	lon = f1['longitude'][:].values
	lowerlatlim=30
	lat = f1['latitude'][:].values
	freq=6
	lowerLatidx=int((90-lowerlatlim)/(lat[0]-lat[1]))
	lonsM=lon
	latsM=lat[0:lowerLatidx]
	xptsM, yptsM=proj(*np.meshgrid(lonsM, latsM))
	u10_main = f1['u10']#.load() # only do this one per month; should be faster hopefully
	v10_main = f1['v10']#.load() # try not loading and see if chunking makes things work?
	print('data loaded')


	for dayT in range(startDay, endDay):
	
		dayStr='%03d' %dayT
		month=np.where(dayT-np.array(monIndex)>=0)[0][-1]
		monStr='%02d' %(month+1)
		dayinmonth=dayT-monIndex[month]

		dayStr='%03d' %dayT

		print('Wind day:', dayT)
		
		#in  kg/m2 per day

		# getting bugs when using function in utils so try just doing directly here

		
		numday=dayinmonth
		# print(lowerLatidx)
		
        # have to use .load for some reason; I think this is memory/time consuming though. but can try I guess
		u10=u10_main[(numday*24):(numday*24)+24:freq, 0:lowerLatidx, :].astype(np.float16)#.load()
		v10=v10_main[(numday*24):(numday*24)+24:freq, 0:lowerLatidx, :].astype(np.float16)#.load()
		WindMag=np.mean(np.sqrt((u10**2)+(v10**2)), axis=0).values
		# mag = 0
		# xpts,ypts = 0,0
		# xptsM, yptsM, lonsM, latsM, WindMag = xpts, ypts, lon, lat, mag
        
		# xptsM, yptsM, lonsM, latsM, WindMag =cF.get_ERA5_wind_days_pyproj(proj, data_path, str(yearT), monStr, dayinmonth, lowerlatlim=30)
		
		# if it's the first day, calculate weights
		if calc_weights == 1:
			# calculate Delaunay triangulation interpolation weightings for first file of the year
			print('calculating interpolation weightings')
			ptM_arr = np.array([xptsM.flatten(),yptsM.flatten()]).T
			tri = Delaunay(ptM_arr) # delaunay triangulation
			calc_weights = 0


		# grid using linearNDInterpolator with triangulation calculated above 
		# (faster than griddata but produces identical output)		
		interp = LinearNDInterpolator(tri,WindMag.flatten())
		windMagG = interp((xptsG,yptsG))

#		cF.plot_gridded_cartopy(lonG, latG, windMagG, proj=ccrs.NorthPolarStereo(central_longitude=-45), out=fig_path+'/ERA5winds'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr, date_string=str(yearT), month_string=str(dayT), extra=extraStr, varStr='ERA5 winds ', units_lab=r'kg/m2', minval=0, maxval=10, cmap_1=plt.cm.viridis)
		
		windMagG.dump(out_path+str(yearT)+'/ERA5winds'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)

#-- run main program
if __name__ == '__main__':
	for y in range(1984, 2020+1, 1):
		print (y)
		main(y, startMonth=4,endMonth=4,data_path=reanalysis_raw_path)
		main(y, startMonth=5,endMonth=5,data_path=reanalysis_raw_path)
		main(y, startMonth=6,endMonth=6,data_path=reanalysis_raw_path)
		main(y, startMonth=7,endMonth=7,data_path=reanalysis_raw_path)





