""" run.py
	
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
from calendar import monthrange

yearS=2012
monthS=8 # September = 8
dayS=0

yearE=2013
monthE=4
#dayE=monthrange(yearE, monthE)[1]

for y in range(2010,2017):
	yearS = y
	yearE = y+1

	dayE=monthrange(yearE, monthE)[1]

	print('yearS = {}'.format(yearS))
	NESOSIM.main(year1=yearS, month1=monthS, day1=dayS, year2=yearE, month2=monthE-1, day2=dayE-1,
	outPathT=model_save_path, 
	forcingPathT=forcing_save_path, 
	figPathT=figure_path+'Model/',
	precipVar='ERA5', windVar='ERA5', driftVar='OSISAF', concVar='CDR', 
	densityTypeT='variable', extraStr='v11', outStr='', IC=2, 
	windPackFactorT=5.8e-7, windPackThreshT=5, leadLossFactorT=5.8e-7,
	dynamicsInc=1, leadlossInc=1, windpackInc=1,scaleCS=False)




