import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spire_data_utils import *
import pickle
import corner
# from pcat_spire import *


# def load_param_dict(timestr, result_path='/Users/richardfeder/Documents/multiband_pcat/spire_results/'):


def gather_posteriors(timestr_list, inject_sz_frac=None, tail_name='6_4_20', band_idx0=0, datatype='real', pdf_or_png='.png'):

	# all_temp_posteriors = []
	figs = []

	band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})

	temp_mock_amps = [None, 0.3, 0.5] # MJy/sr
	flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]



	for i in np.arange(band_idx0, 3):

		all_temp_posteriors = []


		f = plt.figure()

		if inject_sz_frac is not None:
			inject_sz_amp = inject_sz_frac*temp_mock_amps[i]*flux_density_conversion_facs[i]
			print('inject sz amp is ', inject_sz_amp)

		for j, timestr in enumerate(timestr_list):
			gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')

			chain = np.load(filepath+'/chain.npz')

			band=band_dict[gdat.bands[i]]

			if j==0:
			# 	plt.title(band+', $\\langle \\delta F\\rangle = $'+str(np.median(all_temp)))
				if datatype=='real':
					label='Indiv. chains'
				else:
					label='Indiv. mock realizations'

			else:
				label = None

			burn_in = int(gdat.nsamp*gdat.burn_in_frac)

			template_amplitudes = chain['template_amplitudes'][burn_in:, i, 0]

			plt.hist(template_amplitudes, histtype='step', color='k', label=label)

			all_temp_posteriors.extend(template_amplitudes)

		if inject_sz_frac is not None:
			plt.title(band+', $\\langle \\delta F\\rangle = $'+str(np.round(inject_sz_amp - np.median(all_temp_posteriors),6))+' Jy/beam')
		else:
			plt.title(band)


		plt.hist(all_temp_posteriors, label='Aggregate Posterior', histtype='step', bins=15)
		plt.axvline(np.median(all_temp_posteriors), label='Median', linestyle='dashed', color='r')
		plt.axvline(np.percentile(all_temp_posteriors, 16), linestyle='dashed', color='b')
		plt.axvline(np.percentile(all_temp_posteriors, 84), linestyle='dashed', color='b')
		plt.axvline(np.percentile(all_temp_posteriors, 5), linestyle='dashed', color='g')
		plt.axvline(np.percentile(all_temp_posteriors, 95), linestyle='dashed', color='g')

		if inject_sz_frac is not None:
			plt.axvline(inject_sz_amp, label='Injected SZ Amplitude', linestyle='solid', color='k', linewidth='2')

		plt.legend()
		plt.xlabel('Template amplitude [mJy/beam]')
		plt.ylabel('Number of samples')
		plt.savefig('aggregate_posterior_sz_template_injszfrac_'+str(inject_sz_frac)+'_band_'+str(i)+'_'+tail_name+pdf_or_png, bbox_inches='tight')

		figs.append(f)

		plt.close()


	return figs



# timestr_list_no_sz = ['20200511-091740', '20200511-090238', '20200511-085314', '20200511-040620', '20200511-040207', '20200511-035415', '20200510-230221', \
# 							'20200510-230200', '20200510-230147', '20200512-170853', '20200512-170912', '20200512-170920', '20200512-231130', \
# 							'20200512-231147', '20200512-231201', '20200513-041608', '20200513-042104', '20200513-043145', '20200513-112022', \
# 							'20200513-112041', '20200513-112030', '20200513-205034', '20200513-205025', '20200513-205018', '20200514-023951',\
# 							'20200514-024001', '20200514-024009', '20200514-145434', '20200514-145442', \
# 							'20200514-205559', '20200514-205546', '20200514-205608', '20200515-022433', '20200515-023105', '20200515-023151', \
# 							'20200515-075700', '20200515-080337', '20200515-080822']

timestr_list_no_sz = ['20200531-224825', '20200531-224833', '20200531-224841', '20200601-025648', '20200601-030306', '20200601-031630', \
						'20200601-071608', '20200601-071708', '20200601-073958', '20200601-232446', '20200601-232509', '20200601-232518', '20200602-033906', \
						'20200602-035510', '20200602-035940', '20200602-080030', '20200602-081620', '20200602-084239']

timestr_list_sz_0p5 = ['20200603-183737', '20200603-182513', '20200603-182023', '20200603-131227', '20200603-131154', '20200603-131052', '20200604-000848', '20200604-001450', '20200603-235206']

timestr_list_threeband_sz = ['20200607-010226', '20200607-010206', '20200607-010142', '20200605-172052']
timestr_list_twoband_sz = ['20200607-130436', '20200607-130411', '20200607-130345', '20200605-172157']


# fs = gather_posteriors(timestr_list_sz_0p5, inject_sz_frac=0.5)

# fs = gather_posteriors(timestr_list_twoband_sz, tail_name='two_band_sz')


def aggregate_posterior_corner_plot(timestr_list, temp_amplitudes=True, bkgs=True, nsrcs=True, \
							bkg_bands=[0, 1, 2], temp_bands=[0, 1, 2], n_burn_in=1500, plot_contours=True, tail_name='', pdf_or_png='.pdf', nbands=3, save=True):

	flux_density_conversion_facs = dict({0:86.29e-4, 1:16.65e-3, 2:34.52e-3})
	bkg_labels = dict({0:'$B_{250}$', 1:'$B_{350}$', 2:'$B_{500}$'})
	temp_labels = dict({0:'$A_{250}^{SZ}$', 1:'$A_{350}^{SZ}$', 2:'$A_{500}^{SZ}$'})


	nsrc = []
	bkgs = [[] for x in range(nbands)]
	temp_amps = [[] for x in range(nbands)]
	samples = []

	for i, timestr in enumerate(timestr_list):
		chain = np.load('spire_results/'+timestr+'/chain.npz')

		if nsrcs:
			nsrc.extend(chain['n'][n_burn_in:])
		
		for b in bkg_bands:
			if bkgs:
				bkgs[b].extend(chain['bkg'][n_burn_in:,b].ravel()/flux_density_conversion_facs[b])
		
		for b in temp_bands:
			if temp_amplitudes:
				temp_amps[b].extend(chain['template_amplitudes'][n_burn_in:,b].ravel()/flux_density_conversion_facs[b])


	corner_labels = []

	agg_name = ''

	if bkgs:
		agg_name += 'bkg_'
		for b in bkg_bands:
			samples.append(np.array(bkgs[b]))
			corner_labels.append(bkg_labels[b])
	if temp_amplitudes:
		agg_name += 'temp_amp_'
		for b in temp_bands:
			samples.append(np.array(temp_amps[b]))
			corner_labels.append(temp_labels[b])

	if nsrcs:
		agg_name += 'nsrc_'
		samples.append(np.array(nsrc))
		corner_labels.append("N_{src}")


	samples = np.array(samples).transpose()

	print('samples has shape', samples.shape)


	figure = corner.corner(samples, labels=corner_labels, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],plot_contours=plot_contours,
						   show_titles=True,title_fmt='.3f', title_kwargs={"fontsize": 14}, label_kwargs={"fontsize":16})



	if save:
		figure.savefig('corner_plot_'+agg_name+tail_name+pdf_or_png, bbox_inches='tight')

	return figure

def compute_gelman_rubin_diagnostics(list_of_chains):
    
    m = len(list_of_chains)
    n = len(list_of_chains[0])
    
    print('n=',n,' m=',m)
        
    B = (n/(m-1))*np.sum((np.mean(list_of_chains, axis=1)-np.mean(list_of_chains))**2)
    
    W = 0.
    for j in range(m):
        
        sumsq = np.sum((list_of_chains[j]-np.mean(list_of_chains[j]))**2)
        
        W += (1./m)*(1./(n-1.))*sumsq
    
    var_th = ((n-1.)/n)*W + (B/n)
    
    
    Rhat = np.sqrt(var_th/W)
    
    print("rhat = ", Rhat)
    
    return Rhat
    
def compute_chain_rhats(all_chains, labels=['']):
    
    rhats = []
    for chains in all_chains:
        rhat = compute_gelman_rubin_diagnostic(chains)
        
        rhats.append(rhat)
        
    
    f = plt.figure()
    plt.title('Gelman Rubin statistic $\\hat{R}$', fontsize=16)
    plt.bar(labels, rhats, width=0.5, alpha=0.4)
    plt.axhline(1.2, linestyle='dashed', label='$\\hat{R}$=1.2')
    plt.axhline(1.1, linestyle='dashed', label='$\\hat{R}$=1.1')

    plt.legend()
    plt.xticks(fontsize=16)
    plt.ylabel('$\\hat{R}$', fontsize=16)
    plt.show()
    
    return f, rhats



def gather_posteriors_different_amps(timestr_list, label_dict):

	all_temp_posteriors = []
	figs = []


	band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})


	temp_mock_amps = [None, 0.3, 0.5] # MJy/sr
	flux_density_conversion_facs = [86.29e-4, 16.65e-3, 34.52e-3]

	ratios = [0.0, 0.5, 1.0, 1.5]

	temp_mins = [-0.003, -0.012]
	temp_maxs = [0.011, 0.027]


	for i in range(2):

		f = plt.figure(figsize=(12, 4))


		for j, timestr in enumerate(timestr_list):
			gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')

			chain = np.load(filepath+'/chain.npz')

			band=band_dict[gdat.bands[i+1]]

			if j==0:
				linelabel = 'Recovered SZ'
				plt.title('Injected and Recovered SZ amplitudes (RX J1347.5-1145, '+band+')', fontsize=18)
			else:
				linelabel = None
			# if j==0:
			# 	plt.title(band)
			# 	label='Indiv. mock realizations'
			# else:
			# 	label = None

			colors = ['C0', 'C1', 'C2', 'C3']

			burn_in = int(gdat.nsamp*gdat.burn_in_frac)

			template_amplitudes = chain['template_amplitudes'][burn_in:, i+1, 0]

			bins = np.linspace(temp_mins[i], temp_maxs[i], 50)

			plt.hist(template_amplitudes, histtype='stepfilled',alpha=0.7,edgecolor='k', bins=bins,label=linelabel,  color=colors[j], linewidth=2)

			plt.axvline(ratios[j]*temp_mock_amps[i+1]*flux_density_conversion_facs[i+1], color=colors[j], label=label_dict[i+1][j], linestyle='dashed', linewidth=3)

			# all_temp_posteriors.extend(template_amplitudes)


		# plt.hist(all_temp_posteriors, label='Aggregate Posterior', histtype='step', bins=30)

		# plt.axvline(0., label='Injected SZ Amplitude', linestyle='dashed', color='g')

		plt.legend()
		plt.xlabel('Template amplitude [mJy/beam]', fontsize=14)
		plt.ylabel('Number of posterior samples', fontsize=14)
		plt.savefig('recover_injected_sz_template_amps_band_'+str(i+1)+'.pdf', bbox_inches='tight')

		# plt.savefig('aggregate_posterior_sz_template_no_injected_signal_band_'+str(i+1)+'.pdf', bbox_inches='tight')
		plt.show()

		figs.append(f)

	return figs


figs = aggregate_posterior_corner_plot(timestr_list_no_sz, tail_name='testing', temp_bands=[1, 2], nsrcs=False)

# 
# amplitudes = dict({1:[]})
# labels = dict({1:['No SZ signal', '0.15 MJy/sr', '0.3 MJy/sr', '0.45 MJy/sr'], 2:['No signal', '0.25 MJy/sr', '0.5 MJy/sr', '0.75 MJy/sr']})


# timestrs = ['20200510-230147', '20200512-101717', '20200512-101738', '20200512-103101']

# fs = gather_posteriors_different_amps(timestrs, labels)





