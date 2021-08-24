""" run_ioopt.py
	
	run io-optimized nesosim (preload file i/o; for testing)
"""

import matplotlib
matplotlib.use("AGG")
from pylab import *

# do I need all these imports?
import subprocess
import shlex
import sys

from config import forcing_save_path
from config import model_save_path
from config import figure_path

import io_helpers as io

print('Forcing file path:', forcing_save_path)
print('Output path:', model_save_path)
print('Figure save path:', figure_path)


yearS=2011
monthS=8 # August = 7
dayS=0

yearE=2012
monthE=3 # April = 7
dayE=29

# preload data


# load the whole year
month1 = 0
day1 = 0
month2 = 11 #is this the indexing used?, ie would this be december
day2 = 30 # would this be the 31st? I think so

# 
precipVar='ERA5'
windVar='ERA5'
concVar='CDR'
driftVar='OSISAF'
dxStr='100km'
extraStr='v11'

# nesosim adds the dxstr but io load needs this modified 
forcing_io_path=forcing_save_path+dxStr+'/'

# load multiple years

print('loading input data')
forcing_dict = io.load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcing_io_path)

print('finished loading input')


#print(forcing_dict[2012].keys())
#print(forcing_dict[2012]['wind'])


year=2011
#current = forcing_dict[2011]['days']
day=364
#print(len(current))

#print(current[0])
#print(current[-3:])

#print(np.where(current==day)[0][0])

print(yearS, monthS, dayS, yearE, monthE, dayE)


import NESOSIM	
NESOSIM.main(year1=yearS, month1=monthS, day1=dayS, year2=yearE, month2=monthE, day2=dayE,
	outPathT=model_save_path, 
	forcingPathT=forcing_save_path, 
	figPathT=figure_path,
	precipVar='ERA5', windVar='ERA5', driftVar='OSISAF', concVar='CDR', 
	icVar='ERA5', densityTypeT='variable', extraStr='v11', outStr='test_io', IC=2, 
	windPackFactorT=5.8e-7, windPackThreshT=5, leadLossFactorT=2.9e-7,
	dynamicsInc=1, leadlossInc=1, windpackInc=1,scaleCS=True, dx=100000,
	plotdaily=1,forcingVals=forcing_dict)



