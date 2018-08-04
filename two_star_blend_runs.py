import os
import numpy as np
import sys
mock_test_name = 'mock_2star_30'

rewrite = 1



offsets = np.array([1.2], dtype=np.float32)
#offsets = np.array([0.5, 0.6, 0.75, 1.0, 1.2, 1.5], dtype=np.float32)
flux_ratios = np.array([1.0, 2.0, 5.0], dtype=np.float32)
#flux_ratios = np.array([1.0], dtype=np.float32)
r_fluxes = np.array([500.], dtype=np.float32)
#r_fluxes = np.array([250., 500., 1000.], dtype=np.float32)

# cases = ['chain3', 'chain1', 'chain1x3']
cases = ['r+i+g', 'r', 'rx3']
#cases = ['rx3']

if len(sys.argv)>1:
        flu = float(sys.argv[1])
        r_fluxes = np.array([flu], dtype=np.float32)

print 'r_fluxes:', r_fluxes


n_noise_realizations = 3

n_iterations = (len(offsets))*(len(flux_ratios))*(len(r_fluxes))*n_noise_realizations

c = 0

for config_type in cases:
	print 'Working on case', config_type
	for offset in offsets:
		for flux in r_fluxes:
			if config_type=='rx3':
				flux *=3
			for flux_ratio in flux_ratios:
				dataname = mock_test_name+'-' + str(offset)+'-'+str(flux)+'-'+str(flux_ratio)
				print 'Running ' + str(dataname)
				for realization in xrange(n_noise_realizations):
					print(str(c+1) + ' of ' + str(n_iterations))
					c+=1

					if not os.path.isfile('Data/'+mock_test_name+'/'+str(dataname)+'/results/'+config_type+'-'+str(realization+1)+'.npz') or rewrite:
						cmd = 'python pcat.py ' + str(dataname) + ' 0 0 mock2 1 '+str(config_type) + ' ' + str(realization+1)
						print cmd
						print realization+1
						os.system(cmd)
					else:
						print 'did this one already!'
