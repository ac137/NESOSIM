"""
gridERAIsf.py

script for gridding ERA-Interim snowfall data

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

from config import reanalysis_raw_path
from config import forcing_save_path
from config import figure_path



def get_ERAI_precip_days_pyproj(proj, erai_data_path, yearStr, monStr, numday, lowerlatlim=0, varStr='sf'):
	print(erai_data_path+'fcinterim_daily_{}*.grb'.format(yearStr))	
	f1 = xr.open_mfdataset(erai_data_path+'fcinterim_daily_{}*.grb'.format(yearStr),engine='cfgrib')

	# Units given in m of freshwater in a 12 hour period. 
	# So to convert to kg/m2/s multiply by den
	#var=var*1000./(60.*60.*12.)

	lon = f1['longitude'].values[:]


	lat = f1['latitude'].values[:]
	print(lat[1]-lat[0])


	lowerLatidx=int((90-lowerlatlim)/(lat[0]-lat[1]))
	print(lowerLatidx)
	lat=lat[0:lowerLatidx]

	# Had to switch lat and lon around when using proj!
	xpts, ypts=proj(*np.meshgrid(lon, lat))
	print(xpts,ypts)

	# in units of m of water so times by 1000, the density of water, to express this as kg/m2
	# data is every 12-hours, so need to multiply numdays by 2, then also sum over the first two time intervals

	varT=f1[varStr][(numday*2):(numday*2)+2, 0:lowerLatidx, :].astype(np.float16)*1000.
	
	var=np.sum(varT, axis=0).values

	return xpts, ypts, lon, lat, var


def main(year, startMonth=0, endMonth=4, dx=50000, extraStr='v11', data_path=reanalysis_raw_path+'ERAI/', out_path=forcing_save_path+'Precip/ERAI/', fig_path=figure_path+'Precip/ERAI/', anc_data_path='../../anc_data/'):


	xptsG, yptsG, latG, lonG, proj = cF.create_grid()
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

	for dayT in range(startDay, endDay):
	
		dayStr='%03d' %dayT
		month=np.where(dayT-np.array(monIndex)>=0)[0][-1]
		monStr='%02d' %(month+1)
		dayinmonth=dayT-monIndex[month]
		print('Precip day:', dayT, dayinmonth)
		
		#in  kg/m2 per day
		xptsM, yptsM, lonsM, latsM, Precip =get_ERAI_precip_days_pyproj(proj, data_path, str(yearT), monStr, dayinmonth, lowerlatlim=30, varStr=varStr)
		print(Precip)
		PrecipG = griddata((xptsM.flatten(), yptsM.flatten()), Precip.flatten(), (xptsG, yptsG), method='linear')

		cF.plot_gridded_cartopy(lonG, latG, PrecipG, proj=ccrs.NorthPolarStereo(central_longitude=-45), out=fig_path+'/'+varStr+'-'+str(yearT)+'_d'+str(dayT)+'T2', date_string=str(yearT), month_string=str(dayT), extra=extraStr, varStr='ERAI snowfall ', units_lab=r'kg/m2', minval=0, maxval=10, cmap_1=plt.cm.viridis)
		
		PrecipG.dump(out_path+str(yearT)+'/ERAI'+varStr+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)

#-- run main program
if __name__ == '__main__':
	for y in range(2010, 2015+1, 1):
		print (y)
		main(y,data_path='/users/jk/18/acabaj/EI/')


