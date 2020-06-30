# compare OIB snow depths with NESOSIM snow depths

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')
import utils as ut
import scipy.stats as st
from config import forcing_save_path,figure_path,oib_data_path,model_save_path



OIBpath = forcing_save_path + 'OIB/'


# file path construction etc. - fix later to work with different files/settings/etc.

sf='ERA5'
wind='ERA5'
drift='OSISAF'
sic='CDR'
wpf=5.8e-7
wpt=5
llf=5.8e-7

# hardcoding this for now
dirname_nesosim = 'ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_WPF5.8e-07_WPT5_LLF5.8e-07-50kmv11'
fname_nesosim = 'ERA5CSscaledsfERA5windsOSISAFdriftsCDRsicrhovariable_IC2_DYN1_WP1_LL1_WPF5.8e-07_WPT5_LLF5.8e-07-50kmv11-01092018-30042019'


# testing on 2019-04-06
# model started running on day 0 (1st day) of month 8 (september)



# clean up later
outPath = model_save_path
folderStr = dirname_nesosim
dayT = 217 # for 2019-04-06 (testing)
day_val = 217
totalOutStr = fname_nesosim

file_list = os.listdir(OIBpath)
print(file_list)

# selecting specific file corresponding to 2019-04-06
for f in file_list[9:10]:

	fname = OIBpath + f

	date = f[:8]
	print(date)

	# load OIB depth array
	depth_OIB = np.load(fname,allow_pickle=True)

	# plot OIB
	plt.imshow(depth_OIB)
	plt.show()
	print(depth_OIB)

	# count number of non-nan OIB obs (for reference)
	print((~np.isnan(depth_OIB)).sum())



	#get corresponding NESOSIM depth
	snowDepthM=ut.get_budgets2layers_day(['snowDepthTotalConc'], outPath, folderStr, day_val, totalOutStr)
	print(snowDepthM)
	# plot NESOSIM snow depth
	plt.imshow(snowDepthM)
	plt.show()

	# plot of difference
	plt.imshow(snowDepthM - depth_OIB)
	plt.title("NESOSIM - OIB snow depth for 2019-04-06 (m)")
	plt.colorbar()
	plt.show()

	# flatten and select non-nan values for linear regression
	x_val, y_val = np.ravel(depth_OIB),np.ravel(snowDepthM)
	nan_mask= ~np.isnan(x_val) & ~np.isnan(y_val)

	# linear regression, statistics, etc.
	slope,intercept,r_val,p_val,stderr = st.linregress(x_val[nan_mask],y_val[nan_mask])

	print(slope, intercept)
	print('r ',r_val)

	# scatter plot with regression; do kde later? 
	# not many points for single day; do this with all available data as next step
	plt.scatter(x_val,y_val)
	plt.plot(x_val[nan_mask],slope*x_val[nan_mask]+intercept)
	plt.title('NESOSIM vs OIB snow depth for 2019-04-06')
	plt.xlabel('OIB snow depth (m)')
	plt.ylabel('NESOSIM snow depth (m)')
	plt.show()


