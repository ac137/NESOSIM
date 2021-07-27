import numpy as np
import pandas as pd
# import nesosim_OIB_loglike_1par as loglike

import nesosim_OIB_loglike_dens_clim as loglike

# mcmc for single parameter

def write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls):
	'''write output with to an hdf file at location fname'''
	'''note: will overwrite files!'''

	stat_headings = ['r','rmse','merr','std','std_n','std_o']
	valid_df = pd.DataFrame(np.array(stats_list), columns=stat_headings)
	par_arr = np.array(par_list)
	valid_df[par_names[0]] = par_arr[:,0]
	valid_df['loglike'] = loglike_list

	rejected_df = pd.DataFrame(np.array(rejected_stats), columns=stat_headings)
	rej_par_arr = np.array(rejected_pars)
	rejected_df[par_names[0]] = rej_par_arr[:,0]
	rejected_df['loglike'] = rejected_lls

	valid_df.to_hdf(fname, key='valid')
	rejected_df.to_hdf(fname, key='rejected')
	meta_df.to_hdf(fname, key='meta')



# seed for testing
#np.random.seed(42)

# default wpf 5.8e-7
# default llf 2.9e-7 ? different default for multiseason

PAR_NAME = 'WAT' # parameter to vary
ITER_MAX = 10 # start small for testing
UNCERT = 10# obs uncertainty for log-likelihood (also can be used to tune)

PAR_SIGMA = [1] # standard deviation for parameter distribution; can be separate per param
# should be multiplied by 1e-7, but can do that after calculating distribution

# step size determined based on param uncertainty (one per parameter)

# TODO: fix WPF and LLF to work with 3-par possibility later?
if PAR_NAME == 'WPF':
	par_vals = np.array([5.8e-7]) #wpf
	par_names = ['wind packing']
#	P2_DEFAULT = 1.45e-7 # default for other parameter
#	P2_DEFAULT = 2.9e-7
	P2_DEFAULT = 1.79e-7 # optimized default
elif PAR_NAME == 'LLF':
	par_vals = np.array([2.9e-7])# llf default v1.0
#	par_vals = np.array([1.45e-7]) # llf default v1.1
	P2_DEFAULT = 5.8e-7
	par_names = ['blowing snow']

elif PAR_NAME == 'WAT':
	# par_vals = np.array([5.])
	# adjusted so now parameter values can just be passed as a single parameter
	par_vals = np.array([5.0]) # need to keep as array for code to work
	P2_DEFAULT = 5.8e-7 #wind packing
	P3_DEFAULT = 1.79e-7 #blowing snow
	par_names = ['wind action threshold']
	PAR_SIGMA = [0.1]


PARS_INIT = par_vals.copy()


# create a dataframe to save metadata to the files
metadata_headings = ['N_iter','uncert','prior_p1','p2 value','p3 value','sigma_p1', 'oib_prod']
metadata_values = [[ITER_MAX, UNCERT, par_vals[0], P2_DEFAULT, P3_DEFAULT, PAR_SIGMA[0], 'MEDIAN']]
meta_df = pd.DataFrame(metadata_values, columns=metadata_headings)


NPARS = len(par_vals)

print('calculating initial log-likelihood')

# this old function call was for the 1-par loglike
# p0, stats_0 = loglike.main(par_vals, UNCERT, PAR_NAME, [P2_DEFAULT,P3_DEFAULT]) # initial likelihood function

# order of parameters?
# this depends on which parameter is being varied

# might need a few of these if statements; 
# if I do some clever indexing I could be rid of this but I'll leave it for now
if PAR_NAME == 'WAT':
	# order of parameters is always wind packing, blowing snow, wind action
	pars_loglike = [P2_DEFAULT, P3_DEFAULT, par_vals[0]]
#TODO add conditions for WPF, LLF

p0, stats_0 = loglike.main(pars_loglike, UNCERT)
print ('initial setup: params {}, log-likelihood: {}'.format(par_vals, p0))
print('r, rmse, merr, std, std_n, std_o')
print(stats_0)

par_list = [par_vals] # now an nx2 list
loglike_list = [p0]
stats_list = [stats_0] # collect rmse and r also, etc.
#var_cond_list=[]
#diff_list=[]
rejected_pars = []
rejected_lls = []
rejected_stats = []


# first just try metropolis (don't reject proposed values of params)

# steps to take
# np.randon.normal(mean, sigma, shape); sigma can be an array

# maybe change this later to not pre-calculate steps so that
# this doesn't take up space in memory

if PAR_NAME == 'WAT':
	# don't multiply by 1e-7 here since the value is O(1e0)
	step_vals = np.random.normal(0, PAR_SIGMA, (ITER_MAX, NPARS))
else:
	step_vals = np.random.normal(0, PAR_SIGMA, (ITER_MAX, NPARS))*1e-7
# reshape this if the number of params changes
# reject any new parameter less than 0


# open files to store parameters
acceptance_count = 0

# main mcmc loop
for i in range(ITER_MAX):
	print('iterating')
	# random perturbation to step; adjust choice here
	rand_val = step_vals[i]

	# adjust parameters (not checking param distribution for now)
	par_new = par_vals + rand_val

	print('new parameter ', par_new)

	# if any of the parameters are less than zero, don't step there;
	# conversely, if none of the parameters are less than zero, proceed

	# to check param distribution, check the difference of par_new and par_vals
	# not doing this for now

	if (par_new < 0).any() == False:
		print('calculating new log-likelihood')
		# calculate new log-likelihood
		# p, stats = loglike.main(par_new, UNCERT, PAR_NAME, P2_DEFAULT)
		#TODO: add conditions for WPF, LLF
		if PAR_NAME == 'WAT':
			par_new_loglike = [P2_DEFAULT, P3_DEFAULT, par_new[0]]
		p, stats = loglike.main(par_new_loglike, UNCERT)
		# accept/reject; double-check this with mcmc code
		# in log space, p/q becomes p - q, so check difference here
		# checking with respect to uniform distribution
		var_cond = np.log(np.random.rand())
#		var_cond_list.append(var_cond)
#		diff = p-p0
#		diff_list.append(diff)
		if p-p0 > var_cond:
			# accept value
			print('accepted value')
			acceptance_count += 1
#			print('acceptance rate: {}/{} = {}'.format(acceptance_count,i+1,acceptance_count/float(i+1)))
			par_vals = par_new
			p0 = p
			# append to list/ possibly save these to disk so that interrupting the
			# process doesn't lose all the info?
			print('parameters ', par_vals)
			par_list.append(par_vals)
			loglike_list.append(p0)
			stats_list.append(stats)
		else:
			print('rejected value')
			rejected_pars.append(par_new)
			rejected_lls.append(p)
			rejected_stats.append(stats)

		print('acceptance rate: {}/{} = {}'.format(acceptance_count,i+1,acceptance_count/float(i+1)))
	if i>0 and i%1000 == 0:
		# save output every 1k iterations just in case
		print('Writing output for {} iterations...'.format(i))
		# use ITER_MAX to overwrite here, i to create separate files (more disk space but safer)
		fname = 'mcmc_output_1par_i{}_u_{}_p0_{}_{}_pd_{}_{}_s0_{}_noseed.h5'.format(i,UNCERT,PAR_NAME, PARS_INIT[0],P2_DEFAULT, P3_DEFAULT, PAR_SIGMA[0])
		write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls)



# save final output to file
fname = 'mcmc_output_1par_i{}_u_{}_p0_{}_{}_pd_{}_{}_s0_{}_noseed.h5'.format(ITER_MAX,UNCERT,PAR_NAME,PARS_INIT[0],P2_DEFAULT, P3_DEFAULT, PAR_SIGMA[0])
print(ITER_MAX)
print(fname)
write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls)

