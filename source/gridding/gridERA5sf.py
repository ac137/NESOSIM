
""" gridERA5sf.py
	
	Script to grid the ERA5 snowfall data
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

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

from config import reanalysis_raw_path
from config import forcing_save_path
from config import figure_path




def main(year, startMonth=8, endMonth=11, dx=100000, extraStr='v11_1', data_path=reanalysis_raw_path+'ERA5/', out_path=forcing_save_path+'Precip/ERA5/', fig_path=figure_path+'Precip/ERA5/', anc_data_path='../../anc_data/'):


	xptsG, yptsG, latG, lonG, proj = cF.create_grid(dxRes=dx)
	print(xptsG)
	print(yptsG)

	dxStr=str(int(dx/1000))+'km'
	print(dxStr)


	region_mask, xptsI, yptsI = cF.get_region_mask_pyproj(anc_data_path, proj, xypts_return=1)
	region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')

	varStr='sf'

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

	# start with first day for Delaunay triangulation
	dayT0 = startDay
	dayStr='%03d' %dayT0
	month=np.where(dayT0-np.array(monIndex)>=0)[0][-1]
	monStr='%02d' %(month+1)
	dayinmonth=dayT0-monIndex[month]
	print('Precip day:', dayT0, dayinmonth)
	# calculate Delaunay grid for test
	xptsM, yptsM, lonsM, latsM, Precip =cF.get_ERA5_precip_days_pyproj(proj, data_path, str(yearT), monStr, dayinmonth, lowerlatlim=30, varStr=varStr)
	print('calculating Delaunay triangulation')

	ptM_arr = np.array([xptsM.flatten(),yptsM.flatten()]).T
	print('array shape, should be npts,ndim')
	print(ptM_arr.shape)

	tri = Delaunay(ptM_arr)
	# end Delaunay calculation

	for dayT in range(startDay, endDay):
	
		dayStr='%03d' %dayT
		month=np.where(dayT-np.array(monIndex)>=0)[0][-1]
		monStr='%02d' %(month+1)
		dayinmonth=dayT-monIndex[month]
		print('Precip day:', dayT, dayinmonth)
		
		#in  kg/m2 per day
		xptsM, yptsM, lonsM, latsM, Precip =cF.get_ERA5_precip_days_pyproj(proj, data_path, str(yearT), monStr, dayinmonth, lowerlatlim=30, varStr=varStr)
		print(Precip)
# old precip grid code
#		PrecipG = griddata((xptsM.flatten(), yptsM.flatten()), Precip.flatten(), (xptsG, yptsG), method='linear')
# new precip grid code
		interp = LinearNDInterpolator(tri,Precip.flatten())
		PrecipG = interp((xptsG,yptsG))

		cF.plot_gridded_cartopy(lonG, latG, PrecipG, proj=ccrs.NorthPolarStereo(central_longitude=-45), out=fig_path+'/'+varStr+'-'+str(yearT)+'_d'+str(dayT)+'T2', date_string=str(yearT), month_string=str(dayT), extra=extraStr, varStr='ERA5 snowfall ', units_lab=r'kg/m2', minval=0, maxval=10, cmap_1=plt.cm.viridis)
		
		PrecipG.dump(out_path+str(yearT)+'/ERA5'+varStr+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)

#-- run main program
if __name__ == '__main__':
#	data_path_ac = '/users/jk/17/acabaj/e5-daily/'
	data_path_ac = '/data/kushner_group/ERA/achereque/snowfall_full_data/'
	for y in range(1989, 1990, 1):
		print (y)
#		main(y,0,4,dx=100000,data_path=data_path_ac)
#		main(y,7,7,dx=100000,data_path=data_path_ac)

		main(y,3,4,data_path=data_path_ac)



