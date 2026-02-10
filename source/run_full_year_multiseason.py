""" run_full_year_multiseason.py
	
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
#yearS=1992
# yearS=1980
# yearS=2012
#yearE=2020
#yearE=1992
yearE=2023 # year of end date (i.e. ends at e.g. august 2020 if set to 2020 here, 2020 will not run unless set to 2021 here

monthS=8 # August = 7
dayS=0
#yearS=2019

monthE=7 # April = 4
dayE=30 # end day off by 1

# default values
LLF = 2.9e-7
WPF = 5.8e-7
ALF = 2.2e-8

# calibration default values from MCMC 2023 etc
WPF = 2.0504155592128743e-06
LLF = 4.0059442776163867e-07
ALF = LLF*0.15
melt_factor = -4.

# # latest optimal values for 4-param calibration
# WPF = 7.78e-07
# LLF = 7.39e-07
# ALF = 2.20e-08
# melt_factor = -7.52e-01

print(yearS, monthS, dayS, yearE, monthE, dayE)

melt_threshold = 0
melt_method = 'linear'
# melt_method = 'no_melt' # don't need to set this anymore for no-melt because code below takes care of it
weigh_density = True

######### IS MELT HAPPENING:
# if 1 then melt is happening, if 0 melt is not happening
melt_loss_flag=1




melt_method_to_str = {'linear':'lin', 'constant':'const','melt_day_constant':'mday_const',
                      'melt_day_linear':'mday_lin','no_melt':'no_melt'}

melt_str = melt_method_to_str[melt_method]

if weigh_density:
    melt_str = 'denswt_' + melt_str

cs_scaling = True

#output_string = 'denswt_lin_mt_{}_mf_{}'.format(melt_threshold, melt_factor)
#output_string = 'lin_mt_{}_mf_{}'.format(melt_threshold, melt_factor)
#output_string = '{}_mt_{}_mf_{}'.format(melt_str, melt_threshold, melt_factor) + '_noacconmelt'

# THIS IS OVERRIDDEN IF MELT LOSS FLAG IS ZERO
output_string = '{}_mt_{}_mf_{}'.format(melt_str, melt_threshold, melt_factor) + '_noacconmelt'+'_upperfirst'#+'warmprecipthresh'#+'_precipthresh50'
output_string += '_t2mmax' #+ 'test'# denoting use of max daily t2m
#output_string = 'no_melt'
output_string+='_continuoustest'

# 
if melt_loss_flag ==0:
	melt_method = 'no_melt'
	output_string = ''
	print('RUNNING WITHOUT MELT')

#output_string = 'denswt_meltday3_lin_mt_{}_mf_{}'.format(melt_threshold, melt_factor)

IC = 3 # initial conditions
budget_prev = None


for y in range(yearS, yearE):
    # if y == 1987:
    #     continue
    

    budget_prev = NESOSIM.main(year1=y, month1=monthS, day1=dayS, year2=y+1, month2=monthE, day2=dayE,
	outPathT=model_save_path, 
	forcingPathT=forcing_save_path, 
	figPathT=figure_path,
	precipVar='ERA5', windVar='ERA5', driftVar='NSIDCv4', concVar='CDR', 
	icVar='ERA5', densityTypeT='variable', extraStr='v11', outStr=output_string, IC=IC, 
	windPackFactorT=WPF, windPackThreshT=5, leadLossFactorT=LLF, atmLossFactorT=ALF, meltThreshT=melt_threshold, meltFactorT=melt_factor,
	dynamicsInc=1, leadlossInc=1, windpackInc=1,atmlossInc=1,meltlossInc=melt_loss_flag,scaleCS=cs_scaling, dx=100000,
	plotdaily=0, melt_method=melt_method, melt_dens_wt=weigh_density,
							   returnBudget=1, prev_year_budget = budget_prev)

# print(budget_prev['snowDepth'][4,0,:,:])


