# io_helpers.py: functions for loading NESOSIM input into memory to enable
# more efficient running of NESOSIM-MCMC calibration
# by Alex Cabaj, includes code adapted from NESOSIM by Alek Petty

import numpy as np
import numpy.ma as ma

import xarray as xr
import datetime


def loadDay(yearT, dayT, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath):
	""" Load daily forcings
	similar to loadData but added different error handling because this is being used for all days of the year

	"""
	dayStr='%03d' %dayT
	# print('Loading gridded snowfall forcing from:', forcingPath+'Precip/'+precipVar+'/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)

	#------- Read in precipitation -----------
	try:
		# print('Loading gridded snowfall forcing from:', forcingPath+'Precip/'+precipVar+'/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		precipDayG=np.load(forcingPath+'Precip/'+precipVar+'/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr, allow_pickle=True)
#		precipDayG=np.load(forcingPath+'Precip/'+precipVar+'/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr+'_1', allow_pickle=True)

	except:
		if (dayStr=='365'):
			
			precipDayG=np.load(forcingPath+'Precip/'+precipVar+'/sf/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+'364'+extraStr, allow_pickle=True)
			#precipDayG=np.load(forcingPath+'Precip/'+precipVar+'/sf/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+'364'+extraStr+'_1', allow_pickle=True)

			print('no leap year data, used data from the previous day')
		else:
			print('No precip data for {}'.format(dayStr))
			precipDayG = None
			
	
	#------- Read in wind magnitude -----------
	try:
		# print('Loading gridded wind forcing from:', forcingPath+'Winds/'+windVar+'/'+str(yearT)+'/'+windVar+'winds'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		windDayG=np.load(forcingPath+'Winds/'+windVar+'/'+str(yearT)+'/'+windVar+'winds'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr, allow_pickle=True)
		
	except:
		if (dayStr=='365'):
			print('no leap year data, using data from the previous day')
			windDayG=np.load(forcingPath+'Winds/'+windVar+'/'+str(yearT)+'/'+windVar+'winds'+dxStr+'-'+str(yearT)+'_d'+'364'+extraStr, allow_pickle=True)
		
		else:
			print('No precip data for {}'.format(dayStr))
			windDayG = None

	#------- Read in ice concentration -----------
	try:
		print('Loading gridded ice conc forcing from:', forcingPath+'IceConc/'+concVar+'/'+str(yearT)+'/iceConcG_'+concVar+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		iceConcDayG=np.load(forcingPath+'IceConc/'+concVar+'/'+str(yearT)+'/iceConcG_'+concVar+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr, allow_pickle=True)
		
	except:
		if (dayStr=='365'):
			print('no leap year data, using data from the previous day')
			iceConcDayG=np.load(forcingPath+'IceConc/'+concVar+'/'+str(yearT)+'/iceConcG_'+concVar+dxStr+'-'+str(yearT)+'_d'+'364'+extraStr, allow_pickle=True)
	
		else:
			print('No ice conc data for {}'.format(dayStr))
			iceConcDayG = None

	# fill with zero if there's iceconc
	# can't just say 'if iceConcDayG because that tries to evaluate truth of array'
	if type(iceConcDayG) != type(None):
		iceConcDayG[~np.isfinite(iceConcDayG)]=0.
	
	#------- Read in ice drifts -----------
	try:
		print('Loading gridded ice drift forcing from:', forcingPath+'IceDrift/'+driftVar+'/'+str(yearT)+'/'+driftVar+'_driftG'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		driftGdayG=np.load(forcingPath+'IceDrift/'+driftVar+'/'+str(yearT)+'/'+driftVar+'_driftG'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr, allow_pickle=True)	

	except:
		# if no drifts exist for that day then just set drifts to nan array (i.e. no drift).
		print('No drift data')
		if type(iceConcDayG) != type(None):
			driftGdayG = np.empty((2, iceConcDayG.shape[0], iceConcDayG.shape[1]))
			driftGdayG[:] = np.nan
		else:
			driftGdayG = None

	if type(driftGdayG) != type(None):
		driftGdayG = ma.filled(driftGdayG, np.nan)
	#print(driftGdayG)

	return iceConcDayG, precipDayG, driftGdayG, windDayG


def getLeapYr(year):
	leapYrs=[1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]
	if year in leapYrs:
		numDays=366
	else:
		numDays=365
	return numDays


def getDays(year1, month1, day1, year2, month2, day2):
	"""Get days in model time period
	"""
	numDaysYear1=getLeapYr(year1)

	dT01 = datetime.datetime(year1, 1, 1)
	d1 = datetime.datetime(year1, month1+1, day1+1)
	d2 = datetime.datetime(year2, month2+1, day2+1)
	startDayT=(d1 - dT01).days
	numDaysT=(d2 - d1).days+1

	fmt = '%d%m%Y'
	date1Str=d1.strftime(fmt)
	date2Str=d2.strftime(fmt)

	return startDayT, numDaysT, numDaysYear1, date1Str+'-'+date2Str


# would technically be faster to traverse by dataset rather than by day, but
# this will suffice for now

def load_year_into_memory(year1, month1, day1, year2, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath):
	''' load a single year of NESOSIM input data into memory '''

	startDay, numDays, numDaysYear1, dateOut= getDays(year1, month1, day1, year2, month2, day2)
	
	# variables to store values

	iceConcY, precipY, driftGY, windY, days = [],[],[],[],[]

	currentYears = [] # store current year 
	yearCurrent=year1

	for x in range(numDays):	
		day = x+startDay

		currentYears.append(yearCurrent)

		# load single day of data
		iceConcDayG, precipDayG, driftGdayG, windDayG =loadDay(yearCurrent, day, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath)

		# add values to array
		iceConcY.append(iceConcDayG)
		precipY.append(precipDayG)
		driftGY.append(driftGdayG)
		windY.append(windDayG)
		days.append(day)

	return np.array(days), currentYears, iceConcY, precipY, driftGY, windY


def load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath):
	'''Construct a dictionary of data from multiple years'''
	year_dict = {} # make a nested dictionary by year

	for y in range(yearS,yearE+1):
		days, currentYears, iceConcY, precipY, driftGY, windY = load_year_into_memory(y, month1, day1, y, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath)
		d = {}
		d['days']=days 
		d['current_years']=currentYears
		d['iceConc']=iceConcY
		d['precip']=precipY
		d['drift']=driftGY
		d['wind']=windY

		# add the sub-dictionary for the given year to the main dictionary
		year_dict[y] = d

	return year_dict



# ended up copying the read_daily_data_from_memory function directly into
# NESOSIM to avoid import conflicts
def read_daily_data_from_memory(yearT, dayT, year_dict):
	'''Read the daily data from memory. 
	presupposes data is loaded into dictionary year_dict
	where the structure (by layer) is:
	year
	-> 	variable
	->	-> 	data at index by day (of year, not model day)
	'''

	# find index corresponding to year
	current_data = year_dict[yearT]
	# find corresponding day index
	day_idx = np.where(current_data['days']==dayT)[0][0]
	print(day_idx)
	# select data
	iceConcDayG = current_data['iceConc'][day_idx]
	precipDayG = current_data['precip'][day_idx]
	driftGdayG = current_data['drift'][day_idx]
	windDayG = current_data['wind'][day_idx]
	tempDayG = None # placeholder since temp is not needed currently

	return iceConcDayG, precipDayG, driftGdayG, windDayG, tempDayG


if __name__ == '__main__':
	# code for testing

	year = 2019
	day=0
	precipVar='ERA5'
	windVar='ERA5'
	concVar='CDR'
	driftVar='NSIDCv4'
#	driftVar='OSISAF'
	dxStr='50km'
	extraStr='v11'

	print('running')

	# pass this as an argument into loadDay?
	forcingPath = '/home/alex/modeldev/NESOSIM/Forcings/'
	# print('loading years')
	year1 = 2019
	month1 = 0
	day1 = 0
	month2 = 2
	day2 = 2
	year2 = 2019
	
	yearS = 2018
	yearE = 2019

	print('loading multiyear')

	multiyear_data = load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath)

	print(multiyear_data[2018]['days'])


	yearT = 2019
	dayT = 3

	data = read_daily_data_from_memory(yearT, dayT, multiyear_data)

	# output: iceConcDayG, precipDayG, driftGdayG, windDayG, tempDayG
	print(data[4])

