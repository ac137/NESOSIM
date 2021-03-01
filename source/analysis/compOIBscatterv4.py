""" compOIBscatterv4.py


	Script to compare/plot the already binned OIB data to the coincident NESOSIM data
	Model written by Alek Petty (05/22/2019)
	Contact me for questions (alek.a.petty@nasa.gov)

	Input: Binned OIB snow depths, NESOSIM data
	Output: Scatter plots

	Python dependencies:
		See below for the relevant module imports
		Also some function in commongFuncs.py

	Update history:
		05/22/2019: Version 1
		06/11/2019: version 3: excludes values that have too high a difference
					in OIB products
		07/__/2020: version 4: being adapted for NESOSIM v 1.1 by Alex Cabaj

"""

#TODO: clean up imports
import matplotlib
matplotlib.use("AGG")

import numpy as np
from pylab import *
from scipy.io import netcdf
import numpy.ma as ma
from matplotlib import rc
from glob import glob
from scipy.interpolate import griddata
import sys
sys.path.append('../')
import utils as cF
import xarray as xr
import pandas as pd
import os

from config import forcing_save_path,figure_path,oib_data_path,model_save_path


rcParams['ytick.major.size'] = 2
rcParams['axes.linewidth'] = .5
rcParams['lines.linewidth'] = .5
rcParams['patch.linewidth'] = .5
rcParams['ytick.labelsize']=8
rcParams['legend.fontsize']=8
rcParams['font.size']=8
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


def getOIBNESOSIM(dx, folderStr, totalOutStr, yearT, snowType, reanalysis,grid_100=False):#, days_ds, diff_ds):
	"""Grid all the OIB data and correlate"""

	xptsGMall=[]
	yptsGMall=[]
	snowDepthMMall=[]
	snowOIBMall=[]

	# lonG, latG, xptsG, yptsG, nx, ny = cF.getGrid(, dx)
	xptsG, yptsG, latG, lonG, proj = cF.create_grid(dxRes=dx)

	# set up the option to regrid 50x50 nesosim output to 100x100
	if grid_100:
		# when using this flag, set folderStr, totalOutStr to load
		# the 50x50 km resolution product, and set dx=100

		# get 50x50 grid to be regridded to the 100x100 grid
		xptsM, yptsM, latM, lonM, projM = cF.create_grid(dxRes=50000)


	dxStr=str(int(dx/1000))+'km'
	# region_maskG=load(forcingPath+'/Grid/regionMaskG'+dxStr)

	region_mask, xptsI, yptsI = cF.get_region_mask_pyproj(anc_data_pathT, proj, xypts_return=1)
	region_maskG = griddata((xptsI.flatten(), yptsI.flatten()), region_mask.flatten(), (xptsG, yptsG), method='nearest')



	folderPath=forcingPath+'/OIB/{}binned/{}/'.format(dxStr,yearT)
	days_list = os.listdir(folderPath)
	# print(days_list)
	# OIBdaysAfterSepT = days_list

	# iterate over days in the given year for OIB
	# -> could just list days that are from OIB here

	#days_ds - selected days in dataset




	# iterate through available days for year
	# for i, day_val in enumerate(days_ds.values):
	# for day in xrange(size(OIBdaysAfterSepT)):
		# day_val = int(OIBdaysAfterSepT[day][-3:])
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

		print('Num points in day:', size(snowDepthOIB))
		if (size(snowDepthOIB)==0):
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
		maskDay[where(np.isnan(snowDepthOIB))]=1
		# maskDay[where(np.isnan(snowDepthOIB))]
		maskDay[where(snowDepthOIB<=0.04)]=1
		maskDay[where(snowDepthM<=0.04)]=1

		maskDay[where(snowDepthOIB>0.8)]=1
		maskDay[where(snowDepthM>0.8)]=1

		maskDay[where(region_maskG>8.2)]=1


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

# Map projection # change this out
# m = Basemap(projection='npstere',boundinglat=60,lon_0=-45, resolution='l', round=False  )

# clean these variables up later
figpath = figure_path
outPath = model_save_path+'/100km/'
outPath = model_save_path + '/50km/'
print(outPath)
forcingPath = forcing_save_path
anc_data_pathT='../../anc_data/'



#dx=100000# comparing with new model output


#start_years = np.arange(2010,2015)
start_years=[2010]



day_start = 1
month_start = 9
# year_start = 2017
#day_start=15
#month_start=8


icetype=''

# products=['SRLD','JPL','GSFC']
# products_plot=['SRLD','JPL','GSFC','Mean'] # mean = consensus
products = ['GSFC']
products_plot=['GSFC']

#products=['JPL','GSFC','SRLD']

# load file with differences between OIB products (for each day)

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

# Get model grid
dx=100000.
#dxStr='100km' # using this to load NESOSIM at 50km but OIB at 100km


#dx = 50000
dxStr='50km'
extraStr='v11'
#outStr='4x_v2_s03'
outStr='2nov'


folderStr=precipVar+CSstr+'sf'+windVar+'winds'+driftVar+'drifts'+concVar+'sic'+'rho'+densityTypeT+'_IC'+str(IC)+'_DYN'+str(dynamicsInc)+'_WP'+str(windpackInc)+'_LL'+str(leadlossInc)+'_AL'+str(atmlossInc)+'_WPF'+str(windPackFactorT)+'_WPT'+str(windPackThreshT)+'_LLF'+str(leadLossFactorT)+'-'+dxStr+extraStr+outStr

for y in start_years:
	startYear=y
	endYear=y+5
	numYears=endYear-startYear+1
	years=[str(year) for year in range(startYear, endYear+1)]
	years.append('All years')


	snowDepthOIBAllProducts=[]
	snowDepthMMAllProducts=[]

	# iterate over products (SRLD, JPL, GSFC)
	# there's just one here for now; remove loop
	# for i in xrange(size(products)):
	snowDepthOIBAll=[]
	snowDepthMMAll=[]

	# iterate over all years
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
	snowDepthOIBAllProducts.append(snowDepthOIBAll)
	snowDepthMMAllProducts.append(snowDepthMMAll)

	print(snowDepthOIBAll)

	colors=['#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00', 'c', '#a65628', 'k']
	#colors=['m', 'r', 'g', 'b', 'c', 'y', 'k']
	letters=['(a)', '(b)', '(c)','(d)']
	#fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 3.7))
	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3,3.7))
	import scipy.stats as st

	# calculate mean of all products and add to array 
	# [only if more than one product, leave out for now]
	# meanOIB = np.mean(snowDepthOIBAllProducts,axis=0)
	# meanMM = np.mean(snowDepthMMAllProducts,axis=0)
	# snowDepthMMAllProducts.append(list(meanMM))
	# snowDepthOIBAllProducts.append(list(meanOIB))
	# print(meanOIB)


	# print(snowDepthMMAllProducts)
	i=0
	for i in range(size(products_plot)):
		#print ax
		#ax=axs.flatten()[i]
		ax = axs
		sca(ax)

		im1 = scatter(snowDepthMMAllProducts[i],snowDepthOIBAllProducts[i], color='0.4',s=6, marker='x', alpha=0.7, label=products_plot[i])

		# Peform the kernel density estimate
		xx, yy = np.mgrid[0:90:100j, 0:90:100j]
		positions = np.vstack([xx.ravel(), yy.ravel()])
		values = np.vstack([snowDepthMMAllProducts[i], snowDepthOIBAllProducts[i]])
		print(values)
		kernel = st.gaussian_kde(values)
		f = np.reshape(kernel(positions).T, xx.shape)

		cset = ax.contour(xx, yy, f, cmap=cm.Reds)
		# Label plot

		plot(np.arange(0, 100, 0.1), np.arange(0, 100, 0.1), 'k', ls='--')

		#for x in xrange(size(years)):

		#ax.annotate(years[x], xy=(1.015, 0.9-(0.12*x)), xycoords='axes fraction', horizontalalignment='left',color=colors[x])
		ax.annotate(letters[i]+' '+products_plot[i], xy=(0.02, 1.5), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='left',color='k')

		trend, sig, r_a, intercept = cF.correlateVars(snowDepthMMAllProducts[i],snowDepthOIBAllProducts[i])
		r_str = '%.2f' % r_a
		rmse=sqrt(mean((array(snowDepthMMAllProducts[i])-array(snowDepthOIBAllProducts[i]))**2))
		rmsStr='%.0f' % rmse
		merr=np.mean(np.array(snowDepthMMAllProducts[i])-np.array(snowDepthOIBAllProducts[i]))
		merrStr='%.1f' % merr
		std=np.std(np.array(snowDepthMMAllProducts[i])-merr-np.array(snowDepthOIBAllProducts[i]))
		stdStr='%.1f' % std

		ax.annotate('r: '+r_str+'\nRMSE: '+rmsStr+' cm' + '\nMean bias ' + merrStr + ' cm'+'\nSD: '+stdStr+' cm', xy=(0.02, 0.8),
				xycoords='axes fraction', color='k', verticalalignment='bottom', horizontalalignment='left')

		xlim(0, 80)
		ylim(0, 80)
		#if (x>5):

		if (i==0):
			ylabel('OIB snow depth (cm)')
			xlabel('NESOSIM snow depth (cm)')
		if (i==1):
			if SCALED:
				xlabel('NESOSIM ({}-SF) snow depth (cm) with CloudSat scaling; WPF: {}, WPT: {}, LLF: {}'.format(reanalysis,wpfStr,wptStr,llfStr))
			else:
				xlabel('NESOSIM ({}-SF) snow depth (cm); WPF: {}, WPT: {}, LLF: {}'.format(reanalysis,wpfStr,wptStr,llfStr))

	#fig.delaxes(axs.flatten()[8])
	#leg = ax.legend(loc=1, ncol=1,columnspacing=0.3, handletextpad=0.2, bbox_to_anchor=(1.4, 1.), markerscale=2, frameon=False)


	subplots_adjust(bottom=0.11, left=0.2, top = 0.95, right=0.96)
	savefig(figpath+'/NewOIBcorrelationsAllBinned'+folderStr+str(startYear)+str(endYear)+'start'+str(day_start)+str(month_start)+icetype+'Allproductspanels3v2.png', dpi=300)
	#savefig(figpath+'/seasonalSnowDensityComp4'+folderStr+'.png', dpi=300)
	close(fig)
