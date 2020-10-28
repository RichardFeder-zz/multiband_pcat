from pcat_spire import *

base_path = '/Users/luminatech/Documents/multiband_pcat/'
result_path = '/Users/luminatech/Documents/multiband_pcat/spire_results/'


t_filenames = [base_path+'Data/spire/rxj1347_sz_templates/rxj1347_PSW_nr_sze.fits']


cluster_name='rxj1347'

n_fc_terms = 6

init_fc = np.zeros(shape=(n_fc_terms, n_fc_terms, 4))

fmin_levels = [0.05, 0.02, 0.01, 0.007]
nsamps = [50, 200, 200, 500]

cirrus_idx = 5
sim_idx = 200+cirrus_idx
timestr = None
cirrus_path = base_path+'/Data/spire/cirrus_gen/rxj1347_cirrus_sim_idx'+str(cirrus_idx)+'_101920.npz'

median_fcs_iter = []

template_names = ['sze']
template_filename=dict({'sze': t_filenames[0]})
initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.000, 'L':0.014})})

inject_diffuse_comp = True


# for i, fmin in enumerate(fmin_levels):
# 	if i==0:
# 		print('initial fourier coefficients set to zero')
# 		median_fc = init_fc

# 	if i < 2:
# 		viz = False
# 	else:
# 		viz = True

# 	# cirrus_path = base_path+'/Data/spire/cirrus_gen/rxj1347_cirrus_sim_idx'+str(cirrus_idx)+'_101920.npz'
# 	initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.000, 'L':0.014})})

# 	# larger merge/split/birth/death weights at this point? 
# 	ob = lion(band0=0, base_path=base_path, result_path=result_path, bolocam_mask=True, float_background=True, bkg_sig_fac=5.0, bkg_sample_delay=0,\
# 		 cblas=True, openblas=False, visual=viz, show_input_maps=False, tail_name='rxj1347_PSW_sim0'+str(sim_idx)+'_dust',\
# 		  dataname='sim_w_dust', bias=[-0.004], load_state_timestr=timestr, max_nsrc=500, trueminf=fmin, nregion=5, \
# 		   make_post_plots=False, nsamp=nsamps[i], use_mask=True, residual_samples=5, \
# 		    float_fourier_comps=True, init_fourier_coeffs=median_fc, n_fourier_terms=n_fc_terms, show_fc_temps=False, fc_sample_delay=0, fourier_comp_moveweight=200.,\
# 		   float_templates=True, template_names=template_names, template_filename=template_filename, n_frames=3, \
# 		    dfc_prob=1.0, inject_sz_frac=1.0, inject_diffuse_comp=inject_diffuse_comp, diffuse_comp_path=cirrus_path, \
# 		    init_template_amplitude_dicts=initial_template_amplitude_dicts, template_moveweight=0.)

# 	ob.main()
# 	_, filepath, _ = load_param_dict(ob.gdat.timestr)
# 	timestr = ob.gdat.timestr

# 	chain = np.load(filepath+'/chain.npz')
# 	median_fc = np.median(chain['fourier_coeffs'][-10:], axis=0)
# 	median_fcs_iter.append(median_fc)



# final_temp = generate_template(median_fc, n_fc_terms, N=ob.gdat.imsz0[0], M=ob.gdat.imsz0[1])

# np.savez('median_fc_iter'+'_10_21_20_nfcterms='+str(n_fc_terms)+'_'+cluster_name+'_mock_'+str(sim_idx)+'_diffuse_inject.npz', median_fcs_iter=median_fcs_iter)

for iteri in range(3):
	_, filepath, _ = load_param_dict('20201021-142646')

	template_names = ['sze']
	chain = np.load(filepath+'/chain.npz')
	median_fc = np.median(chain['fourier_coeffs'][-10:], axis=0)
	template_filename=dict({'sze': t_filenames[0]})


	flux_density_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})

	dust_I_lams = [3e6, 1.6e6, 8e5] # in Jy/sr
	band_dict = dict({0:'S', 1:'M', 2:'L'})

	dust_rel_SED = [dust_I_lams[i]/dust_I_lams[0] for i in range(len(dust_I_lams))]
	fc_rel_amps = [dust_rel_SED[i]*flux_density_conversion_dict[band_dict[i]]/flux_density_conversion_dict[band_dict[0]] for i in range(len(dust_I_lams))]

	print('fc rel amps aaaaaare ', fc_rel_amps)


	ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, bolocam_mask=True, float_background=True, bkg_sig_fac=5.0, bkg_sample_delay=0,\
		 cblas=True, openblas=False, visual=False, show_input_maps=False, tail_name='rxj1347_PSW_sim0'+str(sim_idx)+'_dust',\
		  dataname='sim_w_dust', bias=[-0.004, -0.007, -0.008], max_nsrc=500, trueminf=0.007, nregion=5, make_post_plots=True, nsamp=2000,\
		   use_mask=True, residual_samples=200, init_fourier_coeffs=median_fc, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, n_frames=3, float_fourier_comps=True, n_fourier_terms=n_fc_terms,\
		    float_templates=True, temp_sample_delay=50, show_fc_temps=False, fc_sample_delay=0, fourier_comp_moveweight=0., dfc_prob=0.0, \
		    inject_diffuse_comp=inject_diffuse_comp, inject_sz_frac=1.0, diffuse_comp_path=cirrus_path, fc_rel_amps=fc_rel_amps, alph=1.0)

	ob.main()

# np.savez('median_fc_iter'+'_10_20_20_nfcterms='+str(n_fc_terms)+'_rxj1347_'+str(sim_idx)+'.npz', median_fcs_iter=median_fcs_iter)
