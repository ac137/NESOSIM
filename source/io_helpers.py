import numpy as np
import xarray as xr
import datetime
# import utils as cF - currently broken on my laptop for some reason
# from NESOSIM import loadData

# don't import this file when running the model because that'll probably lead to 
# some circular imports

# 


#ISSUES: years being iterated over; really should just iterate over years instead of 
# iterating over "NESOSIM days" beause that'll 
# throw the year dictionary indexing off
# i.e. loadData expects to see data sorted by calendar years, not by seasons

# fix this later

#TODO: 
# - better function names
# - testing
# - incorporate this into the main model
# - check for data being None to give exit condition (normally checked by loadData but this isn't being done now)
# - check leap year compliance?





def loadDay(yearT, dayT, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath):
	""" Load daily forcings
	similar to loadData but added different error handling because this is being used for all days of the year

	"""
	dayStr='%03d' %dayT
	# print('Loading gridded snowfall forcing from:', forcingPath+'Precip/'+precipVar+'/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)

	#------- Read in precipitation -----------
	try:
		# print('Loading gridded snowfall forcing from:', forcingPath+'Precip/'+precipVar+'/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		# precipDayG=np.load(forcingPath+'Precip/'+precipVar+'/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr, allow_pickle=True)
		precipDayG=np.load(forcingPath+'Precip/'+precipVar+'/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr+'_1', allow_pickle=True)

	except:
		if (dayStr=='365'):
			
			# precipDayG=np.load(forcingPath+'Precip/'+precipVar+'/sf/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+'364'+extraStr, allow_pickle=True)
			precipDayG=np.load(forcingPath+'Precip/'+precipVar+'/sf/'+str(yearT)+'/'+precipVar+'sf'+dxStr+'-'+str(yearT)+'_d'+'364'+extraStr+'_1', allow_pickle=True)

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
		# print('Loading gridded ice conc forcing from:', forcingPath+'IceConc/'+concVar+'/'+str(yearT)+'/iceConcG_'+concVar+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		iceConcDayG=np.load(forcingPath+'IceConc/'+concVar+'/'+str(yearT)+'/iceConcG_'+concVar+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr+'_n', allow_pickle=True)
		
	except:
		if (dayStr=='365'):
			print('no leap year data, using data from the previous day')
			iceConcDayG=np.load(forcingPath+'IceConc/'+concVar+'/'+str(yearT)+'/iceConcG_'+concVar+dxStr+'-'+str(yearT)+'_d'+'364'+extraStr+'_n', allow_pickle=True)
	
		else:
			print('No ice conc data for {}'.format(dayStr))
			iceConcDayG = None

	# fill with zero if there's iceconc
	# can't just say 'if iceConcDayG because that tries to evaluate truth of array'
	if type(iceConcDayG) == np.ndarray:
		iceConcDayG[~np.isfinite(iceConcDayG)]=0.
	
	#------- Read in ice drifts -----------
	try:
		# print('Loading gridded ice drift forcing from:', forcingPath+'IceDrift/'+driftVar+'/'+str(yearT)+'/'+driftVar+'_driftG'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr)
		driftGdayG=np.load(forcingPath+'IceDrift/'+driftVar+'/'+str(yearT)+'/'+driftVar+'_driftG'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr, allow_pickle=True)	

	except:
		# if no drifts exist for that day then just set drifts to nan array (i.e. no drift).
		print('No drift data')
		if type(iceConcDayG) == np.ndarray:
			driftGdayG = np.empty((2, iceConcDayG.shape[0], iceConcDayG.shape[1]))
			driftGdayG[:] = np.nan
		else:
			driftGdayG = None

	if type(driftGdayG) == np.ndarray:
		driftGdayG = ma.filled(driftGdayG, np.nan)
	#print(driftGdayG)

	#------- Read in temps (not currently used, placeholder) -----------
	# try:
	# 	tempDayG=np.load(forcingPath+'Temp/'+precipVar+'/t2m/'+str(yearT)+'/t2m'+dxStr+'-'+str(yearT)+'_d'+dayStr+extraStr, allow_pickle=True)
	# except:
	# 	# if no drifts exist for that day then just set drifts to masked array (i.e. no drift).
	# 	#print('No temp data')
	# 	tempDayG = np.empty((iceConcDayG.shape[0], iceConcDayG.shape[1]))
	# 	tempDayG[:] = np.nan
	# 	#tempDayG=ma.masked_all((iceConcDayG.shape[0], iceConcDayG.shape[1]))
	
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


# would technically be faster to traverse by dataset rather than by day but
# I'll just go with this for now

def load_year_into_memory(year1, month1, day1, year2, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath):
	# load a single year into memory

	# is year2 needed here?

	startDay, numDays, numDaysYear1, dateOut= getDays(year1, month1, day1, year2, month2, day2)
	
	# iterate over days

	iceConcY, precipY, driftGY, windY = [],[],[],[]

	# array of days
	days = np.arange(numDays-1)+startDay
	currentYears = [] # store current year just in case
	yearCurrent=year1

	for x in range(numDays-1):	
		day = x+startDay
		
		# if (day>=numDaysYear1):
		# 	# If day goes beyond the number of days in initial year, jump to the next year
		# 	day=day-numDaysYear1
		# 	yearCurrent=year2

		currentYears.append(yearCurrent)

		# load single day of data
		# loadData can still do the heavy lifting with filenames

		# need some sort of try/except if there's no data because otherwise it exits
		# I guess I need to write my own load daily data function
		iceConcDayG, precipDayG, driftGdayG, windDayG =loadDay(yearCurrent, day, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath)

		# add values to array
		iceConcY.append(iceConcDayG)
		precipY.append(precipDayG)
		driftGY.append(driftGdayG)
		windY.append(windDayG)

	return days, currentYears, iceConcY, precipY, driftGY, windY


def load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath):
	
	year_dict = {} # do a nested dict

	for y in range(yearS,yearE+1):
		# year_list.append(y)
		days, currentYears, iceConcY, precipY, driftGY, windY = load_year_into_memory(y, month1, day1, y, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath)
		# there might be a cleaner way but this'll do for now
		d = {}
		d['days']=days 
		d['current_years']=currentYears
		d['iceConc']=iceConcY
		d['precip']=precipY
		d['drift']=driftGY
		d['wind']=windY

		year_dict[y] = d
		# double-check that this doesn't do something weird with the assignment in the next loop

	return year_dict



# load_multiple_years could still work okay if we start from January 1st and end on December 31st

# yearCurrent, day, precipVar, windVar, concVar, driftVar, dxStr, extraStr


# what do the vars mean: 
# preipVar is eg. 'ERA5'
# windVar is eg. 'ERA5'
# concVar is eg. 'CDR'
# driftVar is eg. 'OSISAF'
# dxStr is eg. '100km'
# extrastr is eg. 'v2'




# call with the function and store in variables
# data_years = load_multiple_years()
# then afterward call read_data_from_memory

def read_daily_data_from_memory(yearT, dayT, year_dict):
	# presupposes data is loaded into dictionary year_dict
	# where structure is:
	# year
	# -> variable
	# ->-> data at index by day (of year, not model day)



	# is yearT the start year? if it's the current year
	# then that could cause complications
	# alternative to loaddata
	# expet data_years and data_dict_list; output from load_multiple_years
	# (reads values loaded into memory)

	# this'll reach for specific variables

	# select single year and single day


	# find index corresponding to corresponding day; should involve datestart

	# find index corresponding to year
	current_data = year_dict[yearT]
	# ideally you'd want this indexed by day

	# find corresponding day index [double check that dayT corresponds to the
	# day number and not the index, else this is redundant]
	day_idx = np.where(current_data['days']==dayT)[0][0]
	print(day_idx)

	iceConcDayG = current_data['iceConc'][day_idx]
	precipDayG = current_data['precip'][day_idx]
	driftGdayG = current_data['drift'][day_idx]
	windDayG = current_data['wind'][day_idx]
	tempDayG = None # hopefully nothing is expeting anything here!

	# check if any of these are nonetypes? really wish we had case/switch statements



	return iceConcDayG, precipDayG, driftGdayG, windDayG, tempDayG






# test that loadDay works

# vals = loadDay(year, day, precipVar, windVar, concVar, driftVar, dxStr, extraStr)
# print(vals)

# passed; only caveat is that it may need additional suffixes etc.
# depending on what the filenames are formatted as (eg. v11_1 or v11_n etc.)

# test that loading a year works for a few months (valid and invalid days)



if __name__ == '__main__':


	year = 2019
	day=0
	precipVar='ERA5'
	windVar='ERA5'
	concVar='CDR'
	driftVar='OSISAF'
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
	# vals_year = load_year_into_memory(year1, month1, day1, year2, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath)

	# print('done loading years')

	# returns 	return days, currentYears, iceConcY, precipY, driftGY, windY

	# print(vals_year[5])

	# loading years seems to work! extra days just get turned into None
	yearS = 2018
	yearE = 2019

	print('loading multiyear')

	multiyear_data = load_multiple_years(yearS, yearE, month1, day1, month2, day2, precipVar, windVar, concVar, driftVar, dxStr, extraStr, forcingPath)

	print(multiyear_data[2018]['days'])

	# seems to also work! just need to be careful with year1/year2

	yearT = 2019
	dayT = 3

	data = read_daily_data_from_memory(yearT, dayT, multiyear_data)

	# output: iceConcDayG, precipDayG, driftGdayG, windDayG, tempDayG
	print(data[4])

	# seems to all work well enough! 

	# next step:
	# alter nesosim to:
	#	- accept the preloaded data as optional input in the main function
	# 	- unpack the preloaded data properly
	# alter mcmc scripts to to:
	# 	- preload the data in the mcmc main function
	# 	- pass the preloaded data to each iteration of the likelihood function

	# other things to consider preloading:
	# 	- initial conditions
	# 	- cloudsat scaling
	# 	- masks