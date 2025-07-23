""" run_oneseason.py
	
	Run script for the NESOSIM model included in NESOSIM.py 
	Model written by Alek Petty (03/01/2018)
	Contact me for questions (alek.a.petty@nasa.gov) or add a query to the GitHub repo (www.github.com/akpetty/NESOSIM)

	Update history:
		03/01/2018: Version 1

"""

import matplotlib
matplotlib.use("AGG")
from pylab import *


import subprocess
import shlex
import sys

from config import forcing_save_path
from config import model_save_path
from config import figure_path
import NESOSIM


forcing_save_path = '../forcings_full_year/'
# forcing_save_path = r'C:\Users\CabajA\forcings_full_year'
print('Forcing file path:', forcing_save_path)
print('Output path:', model_save_path)
print('Figure save path:', figure_path)

yearS=1980
monthS=8 # August = 7
dayS=0
yearS=2019

yearE=2021
monthE=7 # April = 7
dayE=30 # end day off by 1

# default values
#LLF = 2.9e-7
#WPF = 5.8e-7
#ALF = 2.2e-8


WPF = 2.0504155592128743e-06
LLF = 4.0059442776163867e-07
ALF = LLF*0.15

print(yearS, monthS, dayS, yearE, monthE, dayE)

#melt_factor = -5.
melt_factor = -2.0
melt_threshold = 0
melt_method = 'linear'
#melt_method = 'no_melt'
weigh_density = True

melt_method_to_str = {'linear':'lin', 'constant':'const','melt_day_constant':'mday_const',
                      'melt_day_linear':'mday_lin','no_melt':'no_melt'}

melt_str = melt_method_to_str[melt_method]

if weigh_density:
    melt_str = 'denswt_' + melt_str


#output_string = 'denswt_lin_mt_{}_mf_{}'.format(melt_threshold, melt_factor)
#output_string = 'lin_mt_{}_mf_{}'.format(melt_threshold, melt_factor)
#output_string = '{}_mt_{}_mf_{}'.format(melt_str, melt_threshold, melt_factor) + '_noacconmelt'

output_string = '{}_mt_{}_mf_{}'.format(melt_str, melt_threshold, melt_factor) + '_noacconmelt'+'_upperfirst'#+'_precipthresh'
#output_string = 'no_melt'

#output_string = 'denswt_meltday3_lin_mt_{}_mf_{}'.format(melt_threshold, melt_factor)

for y in range(yearS, yearE):
    if y == 1987:
        continue
    

    NESOSIM.main(year1=y, month1=monthS, day1=dayS, year2=y+1, month2=monthE, day2=dayE,
	outPathT=model_save_path, 
	forcingPathT=forcing_save_path, 
	figPathT=figure_path,
	precipVar='ERA5', windVar='ERA5', driftVar='NSIDCv4', concVar='CDR', 
	icVar='ERA5', densityTypeT='variable', extraStr='v11', outStr=output_string, IC=2, 
	windPackFactorT=WPF, windPackThreshT=5, leadLossFactorT=LLF, atmLossFactorT=ALF, meltThreshT=melt_threshold, meltFactorT=melt_factor,
	dynamicsInc=1, leadlossInc=1, windpackInc=1,atmlossInc=1,meltlossInc=1,scaleCS=True, dx=100000,
	plotdaily=0, melt_method=melt_method, melt_dens_wt=weigh_density)




