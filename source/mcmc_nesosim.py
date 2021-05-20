import numpy as np
import pandas as pd


# use density in mcmc constraints
USE_DENS = True

# is this sort of control flow for import statements reasonable? hopefully
if USE_DENS:
	import nesosim_OIB_loglike_dens as loglike
else:
	import nesosim_OIB_loglike as loglike



def write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls):
	'''write output with to an hdf file at location fname'''
	'''note: will overwrite files!'''

	stat_headings = ['r','rmse','merr','std','std_n','std_o']
	valid_df = pd.DataFrame(np.array(stats_list), columns=stat_headings)
	par_arr = np.array(par_list)
	valid_df[par_names[0]] = par_arr[:,0]
	valid_df[par_names[1]] = par_arr[:,1]
	valid_df['loglike'] = loglike_list

	rejected_df = pd.DataFrame(np.array(rejected_stats), columns=stat_headings)
	rej_par_arr = np.array(rejected_pars)
	rejected_df[par_names[0]] = rej_par_arr[:,0]
	rejected_df[par_names[1]] = rej_par_arr[:,1]
	rejected_df['loglike'] = rejected_lls

	valid_df.to_hdf(fname, key='valid')
	rejected_df.to_hdf(fname, key='rejected')
	meta_df.to_hdf(fname, key='meta')



# seed for testing
#np.random.seed(42)

# default wpf 5.8e-7
# default llf 2.9e-7 ? different default for multiseason

ITER_MAX = 5000# start small for testing
UNCERT = 5 # obs uncertainty for log-likelihood (also can be used to tune)
# par_vals = [1., 1.] #initial parameter values

PAR_SIGMA = [1, 1] # standard deviation for parameter distribution; can be separate per param
# should be multiplied by 1e-7, but can do that after calculating distribution

# step size determined based on param uncertainty (one per parameter)




if USE_DENS:
	DENS_STR = '_density'
else:
	DENS_STR = ''


# try over both wpf and lead loss, now
# order here is [wpf, llf]
#par_vals = np.array([5.8e-7, 2.9e-7])
par_vals = np.array([5.8e-7, 1.45e-7])
PARS_INIT = par_vals.copy()
par_names = ['wind packing', 'blowing snow']

metadata_headings = ['N_iter','uncert','prior_p1','prior_p2','sigma_p1','sigma_p2', 'oib_prod']
metadata_values = [[ITER_MAX, UNCERT, par_vals[0], par_vals[1], PAR_SIGMA[0], PAR_SIGMA[1], 'MEDIAN']]
meta_df = pd.DataFrame(metadata_values, columns=metadata_headings)


NPARS = len(par_vals)

print('calculating initial log-likelihood')
p0, stats_0 = loglike.main(par_vals, UNCERT) # initial likelihood function
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
step_vals = np.random.normal(0, PAR_SIGMA, (ITER_MAX, NPARS))*1e-7
# reshape this if the number of params changes
# reject any new parameter less than 0


# open files to store parameters
acceptance_count = 0

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
		p, stats = loglike.main(par_new, UNCERT)

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
	if i%1000 == 0 and i > 0:
		# save output every 1k iterations just in case
		print('Writing output for {} iterations...'.format(i))
		# use ITER_MAX to overwrite here, i to create separate files (more disk space but safer)
		fname = 'mcmc_output_i{}_u_{}_p0_{}_{}_s0_{}_{}_{}noseed.h5'.format(i,UNCERT,PARS_INIT[0],PARS_INIT[1],PAR_SIGMA[0],PAR_SIGMA[1],DENS_STR)
		write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls)



# save final output to file
fname = 'mcmc_output_i{}_u_{}_p0_{}_{}_s0_{}_{}_{}noseed.h5'.format(ITER_MAX,UNCERT,PARS_INIT[0],PARS_INIT[1],PAR_SIGMA[0],PAR_SIGMA[1],DENS_STR)
print(ITER_MAX)
print(fname)
write_to_file(fname, stats_list, par_list, loglike_list, par_names, rejected_stats, rejected_pars, rejected_lls)

# stat_headings = ['r','rmse','merr','std','std_n','std_o']
# valid_df = pd.DataFrame(np.array(stats_list), columns=stat_headings)
# par_arr = np.array(par_list)
# valid_df[par_names[0]] = par_arr[:,0]
# valid_df[par_names[1]] = par_arr[:,1]
# valid_df['loglike'] = loglike_list

# rejected_df = pd.DataFrame(np.array(rejected_stats), columns=stat_headings)
# rej_par_arr = np.array(rejected_pars)
# rejected_df[par_names[0]] = rej_par_arr[:,0]
# rejected_df[par_names[1]] = rej_par_arr[:,1]
# rejected_df['loglike'] = rejected_lls

# # data I want to keep track of
# # number of iter (can be inferred)
# # uncert
# # which oib product was used
# # parameter sigmas for priors

# fname = 'mcmc_output_i{}_u_{}_p0_{}_{}_s0_{}_{}.h5'.format(ITER_MAX,UNCERT,PARS_INIT[0],PARS_INIT[1],PAR_SIGMA[0],PAR_SIGMA[1])
# # format exponential to 2 decimal places: {:.2e}, in case I need that later
# valid_df.to_hdf(fname, key='valid')
# rejected_df.to_hdf(fname, key='rejected')
# meta_df.to_hdf(fname, key='meta')






# # save like this for now, save more nicely later			
# np.savetxt('par_vals.txt',par_list)
# np.savetxt('log_likelihoods.txt',loglike_list)
# np.savetxt('stat_vals.txt',np.array(stats_list)) 
# #np.savetxt('var_cond.txt',var_cond_list)
# #np.savetxt('diff.txt',diff_list)
# np.savetxt('rejected_pars.txt',rejected_pars)
# np.savetxt('rejected_lls.txt',rejected_lls)
# np.savetxt('rejected_stats.txt',rejected_stats)
