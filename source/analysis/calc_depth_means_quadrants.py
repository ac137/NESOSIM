#
# calculate the seasonal cycle and monthly means of NESOSIM output, by quadrant
#
import matplotlib
matplotlib.use("AGG")


# calculate monthly climatologies for quadrants

import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
#
# import sys
# sys.path.append('../')
# import commonFuncs as cF

# #
# d2 = '/home/alex/NESOSIMdev/Output/scaled_regional_NSIDCv3_ERAI_sf_SICbt_Rhovariable_IC1_DYN1_WP1_LL1_WPF5.8e-07_WPT4_LLF2.9e-07-100kmt2mWindOut/budgets/scaled_regional_NSIDCv3_ERAI_sf_SICbt_Rhovariable_IC1_DYN1_WP1_LL1_WPF5.8e-07_WPT4_LLF2.9e-07-100kmt2mWindOut-15082016-01052017.nc'

WPFs = ['5.8e-07','1.16e-06']
WPTs = ['4','5']
LLFs = ['2.9e-07','5.8e-07']
# select reanalyses without and with scaling; hence layout of lists
REANs = ['ERAI','ERA5','MERRA_2','ERAI','ERA5','MERRA_2']
scalings = ['','','','scaled_regional_lin_','scaled_regional_lin_','scaled_regional_lin_']
MIN_CONC = 0.15 # minimum ice concentration
# scaling = 'scaled_regional_'


day_start=15
month_start=8
# use day_start and month_start eventually also
# params, etc.
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
leadlossInc=0
atmlossInc=1
windPackFactorT=5.8e-7
windPackThreshT=5
#leadLossFactorT=1.16e-6
leadLossFactorT=2.9e-7

# Get model grid
dx=100000.
dxStr='100km' # using this to load NESOSIM at 50km but OIB at 100km
extraStr='v11'
#outStr='4x_v2_s03'
outStr='2nov'


#  testing for now
reanalysis = REANs[1]
wpfStr = WPFs[0]
wptStr = WPTs[1]
llfStr = LLFs[0]

START_YEAR = 2010
END_YEAR = 2015

fig = plt.figure()
ax = plt.gca()

outpath = '/users/jk/18/acabaj/NESOSIM/Output'

# arrays to collect means and standard deviations
means_rean = [[],[],[],[]]
std_rean = [[],[],[],[]]
mon_means_rean = [[],[],[],[]]

# month numbers and names for reindexing

month_nums = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4,5,6]

month_names  =['Jul','Aug','Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr','May','Jun']

if START_YEAR == 1980:
    REANs = ['ERAI','MERRA_2','ERAI','MERRA_2']
    scalings = ['','','scaled_regional_lin_','scaled_regional_lin_']


# quadrant layout for array
# index 0,0 at upper left

#   -90      -45        0
#       CAA   |  GR
#             |  NS
#   -135 ------------- 45
#         ES  |  KA
#         CH  |  LA
#   180      135       90     
# q1: gr/ns, q2: ka/la, q3: es/ch, q4: caa


# iterate thorugh all products (without and with scaling)

quadrant_list=['GRNS','KALA','ESCH','CAA']
# quadrants numbered 1 through 4

# do I need this loop though
for j in range(len(REANs)):
    reanalysis=REANs[j]
    # scaling = scalings[j]


    # folderStr='{}NSIDCv3_{}_sf_SICbt_Rhovariable_IC1_DYN1_WP1_LL1_WPF{}_WPT{}_LLF{}-100kmt2mWindOut'.format(scaling,reanalysis,wpfStr,wptStr,llfStr)

    folderStr=precipVar+CSstr+'sf'+windVar+'winds'+driftVar+'drifts'+concVar+'sic'+'rho'+densityTypeT+'_IC'+str(IC)+'_DYN'+str(dynamicsInc)+'_WP'+str(windpackInc)+'_LL'+str(leadlossInc)+'_AL'+str(atmlossInc)+'_WPF'+str(windPackFactorT)+'_WPT'+str(windPackThreshT)+'_LLF'+str(leadLossFactorT)+'-'+dxStr+extraStr+outStr

    # means for equadrants
    means = [[],[],[],[]]

    # iterate by year
    for year in range(START_YEAR, END_YEAR+1):
        # if year==1987:
        #     continue # skip 1987: missing data

        fn = os.path.join(outpath, folderStr, 'budgets',folderStr+'-1508{}-0105{}.nc'.format(year,year+1))
        print(fn)

        try: # to avoid files that don't exist; this should actually eliminate the need to exclude e5, per se
            df = xr.open_dataset(fn)

            # essentially do the same thing as commonfuncs -> get_budgets_day does, just all at once for efficiency
            # get concentration and depth
            conc = df['iceConc']
            snow_depth = df['snowDepth']

            # array shape: lyrs: 2, day: 250, x: 69, y: 69

            # ice concentration minimum greater than 0.15
            # snow_depth = snow_depth[conc > 0.15] # does this work day by day? just mask after?
            # print(snow_depth)

            # dimension for snow depth: [time, layers, x, y]
            depth_tot = snow_depth[:,0,:,:] + snow_depth[:,1,:,:]
            # print(depth_tot)
            # mask out ice concentration
            depth_tot = depth_tot.where(conc>0.15)

            # print(depth_tot)

            # divide out by ice concentration

            depth_conc = depth_tot / conc

            # need to do some indexing here so that it's done for all four quadrants
            # which quadrant is which here???

            mid = depth_conc.shape[2]//2

            # quadrants in order 1-4 (grns,kala,esch,caa)
            depth_conc_quadrants = [depth_conc[:,:mid,mid:],depth_conc[:,mid:,mid:],depth_conc[:,mid:,:mid],depth_conc[:mid,:mid]]
            # depth_conc_quadrants = [depth_conc[:,mid:,mid:],depth_conc[:,:mid,mid:],depth_conc[:,mid:,:mid],depth_conc[:,:mid,:mid]]
            mean_depth_conc_q = []

            for q in depth_conc_quadrants:
                # calculate mean depth in each quadrant
                mean_depth_conc_q.append(q.mean(dim=['x','y']))


            # mean_depth_conc = depth_conc.mean(dim=['x','y'])
            # print(mean_depth_conc)


            # convert time to datetime

            # day0 = pd.to_datetime()

            dates_depth = pd.date_range('{}-08-15'.format(year), periods=len(mean_depth_conc_q[0]), freq='D')

            for k, quad_val in enumerate(mean_depth_conc_q):

            # regional daily mean; can then collect into months, etc.
            # assign coordinates to time dimensions
                quad_val = quad_val.assign_coords(time=dates_depth)
                means[k].append(quad_val)

            # can assign coordinates to time dimensions
            # means.append(mean_depth_conc_q)#
        except IOError:
            print('file does not exist for year {}'.format(year))
            pass

    # then just aggregate this by month

    df_name = '{}'.format(reanalysis)
    if len(scaling) > 0:
        df_name += ', scaled'

    for i in range(4):
        # iterating over quadrants here

        means[i] = xr.concat(means[i],dim='time').to_dataframe(name=df_name)

        # aggregate by month
        mon_mean = means[i].resample("MS").mean()
   

        print(mon_mean) # how do these look?

        # calculate climatology and spread
        grp = mon_mean.groupby(mon_mean.index.month)
        clim_m = grp.mean()
        clim_std = grp.std()
        # clim_m.plot()
        # plt.show()
        # print(clim_m)

        # reindex by months
        clim_m = clim_m.reindex(month_nums)
        clim_m.index = month_names
        clim_std = clim_std.reindex(month_nums)
        clim_std.index = month_names


        print(clim_m)
        print(clim_std)
        means_rean[i].append(clim_m.copy()) # put copies here to avoid redundancy?
        std_rean[i].append(clim_std.copy())
        mon_means_rean[i].append(mon_mean.copy()) # don't need to reindex this, hopefully?

    # clim_m.plot(marker='x',ax=ax)

# plt.show()
# plt.legend()
# plt.title('Snow depth seasonal cycle comparison')
# plt.savefig('rean_clim_comparison.png')

# create a df


for m in range(4):
    # iterate over quadrants
    # save as hdf files
    mon_m_df = pd.concat(mon_means_rean[m],axis=1)
    mon_m_df.to_hdf('depth_mon_mean_ext_q{}_{}.csv'.format(m+1,quadrant_list[m]))

    clim_df = pd.concat(means_rean[m],axis=1)
    clim_df.to_hdf('depth_clim_2_q{}_{}.csv'.format(m+1, quadrant_list[m]))

    clim_s_df = pd.concat(std_rean[m],axis=1)
    clim_s_df.to_hdf('depth_std_q{}_{}.csv'.format(m+1, quadrant_list[m]))


