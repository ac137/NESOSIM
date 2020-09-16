# compare OIB snow depths with NESOSIM snow depths

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.path.append('../')
import utils as ut
import scipy.stats as st
import cartopy.crs as ccrs

from config import forcing_save_path,figure_path,oib_data_path,model_save_path





# file path construction etc. - fix later to work with different files/settings/etc.

sf='ERA5'
wind='ERA5'
drift='OSISAF'
sic='CDR'
wpf=5.8e-7
wpt=5
llf=5.8e-7
dx=50

day_start = 1
month_start = 9
year_start = 2018

OIBpath = forcing_save_path + 'OIB/{}/'.format(year_start+1)


# hardcoding this for now
#dirname_nesosim = 'ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_WPF5.8e-07_WPT5_LLF5.8e-07-50kmv11'
#fname_nesosim = 'ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_WPF5.8e-07_WPT5_LLF5.8e-07-50kmv11-0109{}-3004{}'.format(year_start,year_start+1)

# dirname_nesosim = 'ERA5sfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_WPF5.8e-07_WPT5_LLF5.8e-07-50kmv11'
# fname_nesosim = 'ERA5sfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_WPF5.8e-07_WPT5_LLF5.8e-07-50kmv11-01092018-30042019'

dirname_nesosim= 'ERA5sfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_WPF5.8e-07_WPT5_LLF1.16e-06-50kmv114x_v2_s03'
fname_nesosim = 'ERA5sfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_WPF5.8e-07_WPT5_LLF1.16e-06-50kmv114x_v2_s03-0109{}-3004{}'.format(year_start,year_start+1)



# testing on 2019-04-06
# model started running on day 0 (1st day) of month 8 (september)



# clean up later
outPath = model_save_path
folderStr = dirname_nesosim

totalOutStr = fname_nesosim

file_list = os.listdir(OIBpath)
print(file_list)

# start day for when NESOSIM was run (9 september)


date_start = pd.to_datetime('{}{:02d}{:02d}'.format(year_start,month_start,day_start))

# selecting specific file corresponding to 2019-04-06
# for f in file_list[9:10]:

OIB_obs_list = []
NESOSIM_list = []

xptsG, yptsG, latG, lonG, proj = ut.create_grid(dxRes=dx)


for f in file_list:
	if f.startswith(str(year_start+1)):
		# select only 2019 for now
		print(f[:8])

		fname = OIBpath + f

		# get number of days from start date
		date_idx = (pd.to_datetime(f[:8])-date_start).days
		print(date_idx)



		# load OIB depth array
		depth_OIB = np.load(fname,allow_pickle=True)

		# plot OIB
		# plt.imshow(depth_OIB)
		# plt.show()
		# print(depth_OIB)

		# count number of non-nan OIB obs (for reference)
		num_OIB_obs =(~np.isnan(depth_OIB)).sum()
		print('number of OIB obs ',num_OIB_obs)

		if num_OIB_obs > 0:

			#get corresponding NESOSIM depth
			snowDepthM=ut.get_budgets2layers_day(['snowDepthTotalConc'], outPath, folderStr, date_idx, totalOutStr)
			# print(snowDepthM)
			# # plot NESOSIM snow depth
			# plt.imshow(snowDepthM)
			# plt.show()

			# # plot of difference
#			plt.imshow(snowDepthM - depth_OIB)
#			plt.imshow(snowDepthM)
#			plt.imshow(depth_OIB)
#			plt.title("NESOSIM and OIB snow depth for {}".format(f[:8]))
#			plt.colorbar()
#			plt.show()
			ut.plot_gridded_cartopy(lonG, latG, snowDepthM-depth_OIB,proj=ccrs.NorthPolarStereo(central_longitude=-45),date_string=f[:8],out='/users/jk/18/acabaj/NESOSIM/Figures/OIB_maps/{}'.format(f[:8]),units_lab='m',varStr='Snow depth difference',minval=-5,maxval=5)

			# mask out values less than/greater than limits

			snowDepthM[snowDepthM<=0.04] = np.nan
			snowDepthM[snowDepthM>0.8] = np.nan
			depth_OIB[depth_OIB<=0.04] = np.nan
			depth_OIB[depth_OIB>0.8] = np.nan


			# flatten and select non-nan values for linear regression
			x_val, y_val = np.ravel(snowDepthM),np.ravel(depth_OIB)
			nan_mask= ~np.isnan(x_val) & ~np.isnan(y_val)


			# print(x_val[nan_mask])


			# check if nonempty

			if len(x_val[nan_mask] > 0):

				# linear regression, statistics, etc.
				slope,intercept,r_val,p_val,stderr = st.linregress(x_val[nan_mask],y_val[nan_mask])

				print('slope and intercept ', slope, intercept)
				# print('r ',r_val)

				# scatter plot with regression; do kde later? 
				# not many points for single day; do this with all available data as next step
				# plt.scatter(x_val,y_val)
				# plt.plot(x_val[nan_mask],slope*x_val[nan_mask]+intercept)
				# plt.title('NESOSIM vs OIB snow depth for 2019-04-06')
				# plt.xlabel('OIB snow depth (m)')
				# plt.ylabel('NESOSIM snow depth (m)')
				# plt.show()

				NESOSIM_list.append(x_val[nan_mask])
				OIB_obs_list.append(y_val[nan_mask])
			else:
				print('no valid values')

# collected all the obs into 2 lists; convert to 1d arrays

OIB_obs_arr = np.concatenate(OIB_obs_list)
NESOSIM_arr = np.concatenate(NESOSIM_list)

rmse = np.sqrt(np.mean((OIB_obs_arr - NESOSIM_arr)**2))

slope,intercept,r_val,p_val,stderr = st.linregress(NESOSIM_arr,OIB_obs_arr)

#x_plot = np.arange(0,0.85,0.1)

print(slope, intercept, r_val)
print(rmse)
plt.scatter(NESOSIM_arr,OIB_obs_arr)
#plt.plot(x_plot,slope*x_plot+intercept)

plt.plot(NESOSIM_arr,slope*NESOSIM_arr+intercept)


plt.text(0.1,0.1,'r = {:01.2f} \n RMSE = {:01.2f}'.format(r_val, rmse))
plt.xlabel('NESOSIM snow depth')
plt.ylabel('OIB snow depth')
plt.title('NESOSIM vs OIB for {}'.format(year_start+1))
#plt.show()
plt.savefig('nesosim_oib_comp_{}_newdrift'.format(year_start+1))
plt.close()


