import numpy as np
import nesosim_OIB_loglike as loglike


# seed for testing
np.random.seed(42)

# default wpf 5.8e-7
# default llf 2.9e-7 ? different default for multiseason

ITER_MAX = 10 # start small
UNCERT = 200 # obs uncertainty for log-likelihood (also can be used to tune)
# par_vals = [1., 1.] #initial parameter values

PAR_SIGMA = 1 # standard deviation for parameter distribution
# should be multiplied by 1e-7, but can do that after calculating distribution

# step size determined based on param uncertainty (one per parameter)

# currently just iterating over lead loss
par_vals = np.array([2.9e-7])
p0, stats_0 = loglike.main(par_vals, UNCERT) # initial likelihood function
print ('initial setup: params {}, log-likelihood: {}'.format(par_vals, p0))
print('r, rmse, merr, std, std_n, std_o')
print(stats_0)

par_list = [par_vals]
loglike_list = [p0]
stats_list = [stats_0] # collect rmse and r also, etc.

# first just try metropolis (don't reject proposed values of params)

# steps to take
step_vals = np.random.normal(0,PAR_SIGMA,ITER_MAX)*1e-7
# reshape this if the number of params changes
# reject any new parameter less than 0


# open files to store parameters
acceptance_count = 0

for i in range(ITER_MAX):
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
		# calculate new log-likelihood
		p, stats = loglike.main(par_new, UNCERT)

		# accept/reject; double-check this with mcmc code
		# in log space, p/q becomes p - q, so check difference here
		# checking with respect to uniform distribution
		if p-p0 > np.log(np.random.rand()):
			# accept value
			print('accepted value')
			acceptance_count += 1
			print('acceptance rate: {}/{} = {}'.format(acceptance_count,i,acceptance_count/i))
			par_vals = par_new
			p0 = p
			# append to list/ possibly save these to disk so that interrupting the
			# process doesn't lose all the info?
			print('parameters ', par_vals)
			par_list.append(par_vals)
			loglike_list.append(p0)
			stats_list.append(stats)

# save like this for now, save more nicely later			
np.savetxt('par_vals.txt',par_list)
np.savetxt('log_likelihoods.txt',loglike_list)
np.savetxt('stat_vals.txt',np.array(stats_list)) 

