
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

from config import reanalysis_raw_path
from config import forcing_save_path
from config import figure_path



# my files have this stored in single days; 
# just keep the function here for now so as not to have to change all of utils
def get_ERA5_wind(proj, era5_data_path, yearStr, monStr, numday, freq=6, lowerlatlim=0):

	print(yearStr, monStr, numday)

	f1 = Dataset(era5_data_path+'/ERA5/t2m_uv_pr/e5_t2m_uv_pr_daily_{}.nc'.format(yearStr), 'r')

	lon = f1.variables['longitude'][:]
	
	lat = f1.variables['latitude'][:]
	
	#print lat[1]-lat[0]
	lowerLatidx=int((90-lowerlatlim)/(lat[0]-lat[1]))
	#print lowerLatidx
	lat=lat[0:lowerLatidx]

	# Had to switch lat and lon around when using proj!
	xpts, ypts=proj(*np.meshgrid(lon, lat))

	# data is daily averaged here

	u10=f1.variables['u10'][numday:numday+1, 0:lowerLatidx, :].astype(np.float16)
	v10=f1.variables['v10'][numday:numday+1:, 0:lowerLatidx, :].astype(np.float16)
	mag=np.mean(np.sqrt((u10**2)+(v10**2)), axis=0)


	return xpts, ypts, lon, lat, mag


def main(year, startMonth=0, endMonth=11, dx=50000, extraStr='v11', data_path=reanalysis_raw_path+'ERA5/', out_path=forcing_save_path+'Winds/ERA5/', fig_path=figure_path+'Winds/ERA5/', anc_data_path='../../anc_data/'):


	xptsG, yptsG, latG, lonG, proj = cF.create_grid()
	print(xptsG)
	print(yptsG)

	dxStr=str(int(dx/1000))+'km'
	print(dxStr)


	region_mask, xptsI, yptsI = cF.get_region_mask_pyproj(anc_data_path, proj, xypts_return=1)
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

	for dayT in range(startDay, endDay):
	
		dayStr='%03d' %dayT
		month=np.where(dayT-np.array(monIndex)>=0)[0][-1]
		monStr='%02d' %(month+1)
		dayinmonth=dayT-monIndex[month]

		dayStr='%03d' %dayT

		print('Wind day:', dayT)
		
		#in  kg/m2 per day
		xptsM, yptsM, lonsM, latsM, WindMag =cF.get_ERA5_wind_days_pyproj(proj, data_path, str(yearT), monStr, dayinmonth, lowerlatlim=30)

		windMagG = griddata((xptsM.flatten(), yptsM.flatten()), WindMag.flatten(), (xptsG, yptsG), method='linear')

		cF.plot_gridded_cartopy(lonG, latG, windMagG, proj=ccrs.NorthPolarStereo(central_longitude=-45), out=fig_path+'/ERA5winds-'+str(yearT)+'_d'+str(dayT)+extraStr, date_string=str(yearT), month_string=str(dayT), extra=extraStr, varStr='ERA5 winds ', units_lab=r'kg/m2', minval=0, maxval=10, cmap_1=plt.cm.viridis)
		
		windMagG.dump(out_path+str(yearT)+'/ERA5winds'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)

#-- run main program
if __name__ == '__main__':
	for y in range(2019, 2020+1, 1):
		print (y)
		main(y, startMonth=0,endMonth=4)



