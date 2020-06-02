import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spire_data_utils import *
import pickle
from pcat_spire import *


# def load_param_dict(timestr, result_path='/Users/richardfeder/Documents/multiband_pcat/spire_results/'):


def gather_posteriors(timestr_list):

	all_temp_posteriors = []
	figs = []

	band_dict = dict({0:'250 micron', 1:'350 micron', 2:'500 micron'})


	for i in range(2):

		f = plt.figure()


		for j, timestr in enumerate(timestr_list):
			gdat, filepath, result_path = load_param_dict(timestr, result_path='spire_results/')

			chain = np.load(filepath+'/chain.npz')

			band=band_dict[gdat.bands[i+1]]

			if j==0:
				plt.title(band)
				label='Indiv. mock realizations'
			else:
				label = None

			burn_in = int(gdat.nsamp*gdat.burn_in_frac)

			template_amplitudes = chain['template_amplitudes'][burn_in:, i+1, 0]

			plt.hist(template_amplitudes, histtype='step', color='k', label=label)

			all_temp_posteriors.extend(template_amplitudes)


		plt.hist(all_temp_posteriors, label='Aggregate Posterior', histtype='step', bins=30)
		plt.axvline(np.median(all_temp_posteriors), label='Median', linestyle='dashed', color='r')
		plt.axvline(np.percentile(all_temp_posteriors, 16), linestyle='dashed', color='b')
		plt.axvline(np.percentile(all_temp_posteriors, 84), linestyle='dashed', color='b')
		plt.axvline(np.percentile(all_temp_posteriors, 5), linestyle='dashed', color='g')
		plt.axvline(np.percentile(all_temp_posteriors, 95), linestyle='dashed', color='g')


		plt.axvline(0., label='Injected SZ Amplitude', linestyle='solid', color='k', linewidth='2')

		plt.legend()
		plt.xlabel('Template amplitude [mJy/beam]')
		plt.ylabel('Number of samples')
		plt.savefig('aggregate_posterior_sz_template_no_injected_signal_band_'+str(i+1)+'_6_1_20.pdf', bbox_inches='tight')
		plt.show()

		figs.append(f)

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
						'20200602-035510', '20200602-035940', '20200602-080030', '20200602-081620']

fs = gather_posteriors(timestr_list_no_sz)


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
# 
# amplitudes = dict({1:[]})
# labels = dict({1:['No SZ signal', '0.15 MJy/sr', '0.3 MJy/sr', '0.45 MJy/sr'], 2:['No signal', '0.25 MJy/sr', '0.5 MJy/sr', '0.75 MJy/sr']})


# timestrs = ['20200510-230147', '20200512-101717', '20200512-101738', '20200512-103101']

# fs = gather_posteriors_different_amps(timestrs, labels)





