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

print('Forcing file path:', forcing_save_path)
print('Output path:', model_save_path)
print('Figure save path:', figure_path)

#yearS=2018
yearS = 2010
monthS=8 # August = 7
dayS=0

#yearE=2019
yearE = 2011
monthE=3 # April = 3
dayE=29

print(yearS, monthS, dayS, yearE, monthE, dayE)

# 2par dens-clim default weight
#WPF = 4.471346291064076e-06
#LLF = 8.100961390478088e-07
#WAT = 5
#ICF = 1
#outStr = 'clim_w_2par'
# 3par dens-clim default weight
#WPF = 5.800556949324572e-06
#LLF = 1.1940103584298072e-06
#WAT = 1.5952473002521215
#ICF = 1
#outStr = 'clim_w_3par'

#3par IC
#WPF = 4.478724936158982e-06
#LLF = 2.02291526840765e-06
#WAT = 5
#ICF = 2.818993994204221
#outStr = 'clim_w_3par_ic'

# N_OIB 2par (test)

#WPF = 1.8494718202551285e-06 
#LLF = 2.6153025979938054e-07
#WAT = 5
#ICF = 1
#outStr = 'clim_w_NOIB_2par'

# 0.5 N_OIB 3par

#WPF = 1.0756942568707031e-06
#LLF = 2.904614517650822e-07
#WAT = 0.767370398173719
#ICF = 1
#outStr = 'clim_w_0.5NOIB_3par'


# w1 v1 buoy (weight 1 on bth density and depth) with 2 pars

#WPF = 4.231235446384845e-06
#LLF = 7.65986492677039e-07
#WAT = 5
#ICF = 1
#outStr = 'clim_w_1_buoy_w_1_2par'

# w4v4 buoy (weight 4*default) with 2 pars
# may not have converged yet (caveat)
#WPF = 2.014929250077817e-06
#LLF = 3.949603343869248e-07
#WAT = 5
#ICF = 1
#outStr = 'clim_w_4_buoy_w_4_2par'

# no OIB
#WPF = 1.6264714628471135e-06
#LLF = 1.1560580275760527e-07
#WAT = 5
#ICF = 1
#outStr = 'clim_w_4_buoy_w_4_NO_OIB'

# w1, v1, oib uncert. = 10 cm
#WPF = 2.049653558530976e-06
#LLF = 4.005362127700446e-07
#WAT = 5
#ICF = 1
#outStr = 'clim_w_1_buoy_w_1_oib_uncert_10'

# w4, v4, oib uncert = 10 cm
#WPF = 1.6430798439213618e-06
#LLF = 2.8008671029129494e-07
#WAT = 5
#ICF = 1
#outStr = 'clim_w_4_buoy_w_4_oib_uncert_10'

# oib clim in loglike, w1, v1, u_oib = 10 cm
#WPF = 1.6321262995790887e-06
#LLF =  1.1584399852081886e-07
#WAT = 5
#ICF = 1
#outStr = 'oib_clim_w_1_buoy_v_1_oib_uncert_10'

# 2par, u=10, oib grid only (no buoy or station)
#WPF = 4.851660817621353e-06
#LLF = 2.589963537823198e-06
#WAT = 5
#ICF = 1
#outStr = '2par_oib_grid_only_u10'

# oib averaged final 5k

#WPF = 1.7284668037515452e-06
#LLF = 1.2174787315012357e-07
#WAT = 5
#ICF = 1
#outStr = '2par_oib_averaged_final_5k'

# 3par ic oib detailed

#WPF = 1.0424574017128326e-06
#LLF = 5.32239712044108e-07
#WAT = 5
#ICF = 2.812296039399609
#outStr = '3par_oib_detailed_ic'


# 3par ic oib averaged

#WPF = 5.77125249688052e-07
#LLF = 3.500788217903482e-07
#WAT = 5
#ICF = 6.700714491498554
#outStr = '3par_oib_averaged_ic'

# 2par oib detailed final 5k
#WPF = 2.0504155592128743e-06
#LLF = 4.0059442776163867e-07
#WAT = 5
#ICF = 1
#outStr = '2par_oib_detailed_final_5k'


# 3par ic oib averaged with ic constraint on loglike (0.5% uncert)
#WPF = 2.3450925692135826e-06
#LLF = 1.5380250062998322e-07
#WAT = 5
#ICF = 0.5312831368932197
#outStr = '3par_oib_averaged_ic_with_ic_loglike'

#3par ic oib detailed with ic constraint on loglike (0.5% uncert)
WPF = 2.054536733591188e-06
LLF = 4.0082408272293277e-07
WAT = 5
ICF = 1.0075085112646933
outStr = '3par_oib_detailed_ic_with_ic_loglike_fixed'

# model default
#WPF = 5.8e-07
#LLF = 2.9e-07
#WAT = 5
#ICF = 1
#outStr = 'mcmc'

import NESOSIM	
NESOSIM.main(year1=yearS, month1=monthS, day1=dayS, year2=yearE, month2=monthE, day2=dayE,
	outPathT=model_save_path, 
	forcingPathT=forcing_save_path, 
	figPathT=figure_path,
	precipVar='ERA5', windVar='ERA5', driftVar='OSISAF', concVar='CDR', 
	icVar='ERA5', densityTypeT='variable', extraStr='v11', outStr=outStr, IC=2, 
	windPackFactorT=WPF, windPackThreshT=WAT, leadLossFactorT=LLF,
	dynamicsInc=1, leadlossInc=1, windpackInc=1, atmlossInc=1,scaleCS=True, dx=100000,
	plotdaily=1, ICfactor=ICF)




