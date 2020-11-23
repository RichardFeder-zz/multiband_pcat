from pcat_spire import *
from diffuse_gen import *

''' this script is for testing the iterative background estimation suggested by Mike. the main idea is to 
gradually reduce the minimum flux threshold of the cataloger while fitting fourier coefficients to observations at 
250 micron. as Fmin gets reduced, the fourier coefficients shift as they do not need to model previously sub-Fmin sources.
After a few iterations, the best fit template is saved and used to model the background of all three bands. once the multiband fitting begins, 
the colors of the background model are fixed and so are its fourier coefficients.'''

class pcat_test_suite():


	flux_density_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})
	# in Jy/sr 
	dust_I_lams=dict({'S':3e6, 'M': 1.6e6,'L':8e5})

	# this is the band convention I use almost exclusively throughout the code when there are stray indices
	band_dict = dict({0:'S', 1:'M', 2:'L'})
	
	def __init__(self,\
				base_path='/Users/luminatech/Documents/multiband_pcat/',\
				result_path='/Users/luminatech/Documents/multiband_pcat/spire_results/', \
				cluster_name='rxj1347',\
				sz_tail_name='rxj1347_PSW_nr_sze', \
				cblas=True, \
				openblas=False):
		self.base_path = base_path
		self.result_path = result_path
		self.cluster_name=cluster_name
		self.sz_tail_name = sz_tail_name
		self.cblas=cblas
		self.openblas=openblas

		self.sz_filename = self.base_path+'Data/spire/'+cluster_name+'_sz_templates/'+sz_tail_name+'.fits'


	def procure_cirrus_realizations(self, nsims, cirrus_tail_name='rxj1347_cirrus_sim_idx_', dataname='new_gen3_sims', \
					tail_name='rxj1347_PSW_sim0300'):

		
		tail_name_M = tail_name.replace('PSW', 'PMW')
		tail_name_L = tail_name.replace('PSW', 'PLW')

		planck_s = fits.open(self.base_path+'/Data/spire/'+dataname+'/'+tail_name+'.fits')['PLANCK'].data
		planck_m = fits.open(self.base_path+'/Data/spire/'+dataname+'/'+tail_name_M+'.fits')['PLANCK'].data
		planck_l = fits.open(self.base_path+'/Data/spire/'+dataname+'/'+tail_name_L+'.fits')['PLANCK'].data

		shapes = [planck_s.shape, planck_m.shape, planck_l.shape]
		imdims_max = [np.max(planck_s.shape), np.max(planck_m.shape), np.max(planck_l.shape)]
		imdims_min = [np.min(planck_s.shape), np.min(planck_m.shape), np.min(planck_l.shape)]

		allo = generate_spire_cirrus_realizations(nsims, planck_s, imdims=imdims_max, show=False)
		allo = [[allo[i][b][:shapes[b][0],:shapes[b][1]] for b in range(3)] for i in range(len(allo))]

		for i, al in enumerate(allo):
			print(np.std(al[0]), np.std(al[1]), np.std(al[2]))
			if not os.path.exists(self.base_path+'/Data/spire/cirrus_gen'):
				os.makedirs(self.base_path+'/Data/spire/cirrus_gen')
			np.savez(self.base_path+'/Data/spire/cirrus_gen/'+cirrus_tail_name+str(i)+'.npz', S=al[0], M=al[1], L=al[2])
		print('we are done now!')

	def run_sims_with_injected_sz(self, visual=False, show_input_maps=False, fmin=0.007, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext', \
				      template_names=['sze'], bias=[-0.006, -0.008, -0.01], use_mask=True, max_nsrc=1000, make_post_plots=True, \
				      nsamp=2000, residual_samples=200, inject_sz_frac=1.0):

		# this assumes the SZ template is within the fits data struct 

		initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.001, 'L':0.018})})

		ob = lion(band0=0, band1=1, band2=2, base_path=self.base_path, result_path=self.result_path, burn_in_frac=0.7, float_background=True, \
			  bkg_sample_delay=0, temp_sample_delay=50, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, \
			  float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, \
			  tail_name=tail_name, dataname=dataname, bias=bias, use_mask=use_mask, max_nsrc=max_nsrc, trueminf=fmin, nregion=5, \
			  make_post_plots=make_post_plots, nsamp=nsamp, residual_samples=residual_samples, inject_sz_frac=inject_sz_frac, template_moveweight=40.)

		ob.main()

	def iter_fourier_comps(self, n_fc_terms=10, fmin_levels=[0.05, 0.02, 0.01, 0.007], final_fmin=0.007, \
							nsamps=[50, 100, 200, 500], final_nsamp=2000, \
							template_names=['sze'], nlast_fc=20, dataname='rxj1347_831', tail_name='rxj1347_PSW_nr_1_ext',\
							bias=[-0.004, -0.007, -0.008], max_nsrc=1000, visual=False, alph=1.0, show_input_maps=False, \
							inject_sz_frac=1.0, residual_samples=200, external_sz_file=False, timestr_list_file=None):

		nbands = len(bias)

		if residual_samples > final_nsamp:
			residual_samples = final_nsamp // 2
			print('residual samples changed to', residual_samples)

		if external_sz_file:
			template_filename=dict({'sze': self.sz_filename})
		else:
			template_filename = None
		timestr = None

		initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.001, 'L':0.018})})

		init_fc = np.zeros(shape=(n_fc_terms, n_fc_terms, 4))

		for i, fmin in enumerate(fmin_levels):
			if i==0:
				print('initial fourier coefficients set to zero')
				median_fc = init_fc

			# start with 250 micron image only
			ob = lion(band0=0, base_path=self.base_path, result_path=self.result_path, round_up_or_down='down', bolocam_mask=False, \
						float_background=True, burn_in_frac=0.75, bkg_sample_delay=0, float_templates=True, template_moveweight=0.,\
		 				cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps, init_template_amplitude_dicts=initial_template_amplitude_dicts,\
		 				template_names=template_names, template_filename=template_filename, tail_name=tail_name, dataname=dataname, bias=[bias[0]], load_state_timestr=timestr, max_nsrc=max_nsrc,\
		 				trueminf=fmin, nregion=5, make_post_plots=False, nsamp=nsamps[i], use_mask=True,\
		 				residual_samples=5, init_fourier_coeffs=median_fc, n_frames=3, float_fourier_comps=True, \
		 				n_fourier_terms=n_fc_terms, show_fc_temps=False, fc_sample_delay=0, fourier_comp_moveweight=200.,\
		 				alph=alph, dfc_prob=1.0, inject_sz_frac=1.0)

			ob.main()

			_, filepath, _ = load_param_dict(ob.gdat.timestr, result_path=self.result_path)
			timestr = ob.gdat.timestr

		# use filepath from the last iteration to load estiamte of median background estimate
		chain = np.load(filepath+'/chain.npz')
		median_fc = np.median(chain['fourier_coeffs'][-nlast_fc:], axis=0)


		dust_rel_SED = [self.dust_I_lams[self.band_dict[i]]/self.dust_I_lams[self.band_dict[0]] for i in range(nbands)]
		fc_rel_amps = [dust_rel_SED[i]*self.flux_density_conversion_dict[self.band_dict[i]]/self.flux_density_conversion_dict[self.band_dict[0]] for i in range(nbands)]

		ob = lion(band0=0, band1=1, band2=2, base_path=self.base_path, result_path=self.result_path, round_up_or_down='down', \
					bolocam_mask=False, float_background=True, burn_in_frac=0.75, bkg_sample_delay=0,\
					temp_sample_delay=50, cblas=self.cblas, openblas=self.openblas, visual=visual, show_input_maps=show_input_maps,\
					float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, \
					tail_name=tail_name, dataname=dataname, bias=bias, max_nsrc=max_nsrc, \
					init_fourier_coeffs=median_fc, template_filename=template_filename, trueminf=final_fmin, nregion=5, \
				    make_post_plots=False, nsamp=final_nsamp, use_mask=True, residual_samples=residual_samples, \
				    float_fourier_comps=True, show_fc_temps=False, n_fourier_terms=n_fc_terms, fc_sample_delay=0, \
				    fourier_comp_moveweight=0., movestar_sample_delay=0, merge_split_sample_delay=0, birth_death_sample_delay=0,\
				    alph=alph, dfc_prob=0.0, fc_rel_amps=fc_rel_amps, inject_sz_frac=1.0, timestr_list_file=timestr_list_file)

		ob.main()



# pcat_test = pcat_test_suite(cluster_name='rxj1347')
# pcat_test.iter_fourier_comps(nsamps=[10, 10, 10, 10], final_nsamp=100, visual=True, show_input_maps=True)

base_path='/home/mbzsps/multiband_pcat/'
result_path='/home/mbzsps/multiband_pcat/spire_results/'

timestr_list_file='lensed_no_dust_gen3sims_rxj1347_11_10_20_timestrs.npz'
started_idx_list_file = 'lensed_no_dust_gen3sims_rxj1347_11_10_20_simidxs.npz' 

pcat_test_sim = pcat_test_suite(cluster_name='rxj1347', base_path=base_path, result_path=result_path, cblas=False, openblas=False)
#pcat_test_sim.run_sims_with_injected_sz(dataname='new_gen3_sims', tail_name='rxj1347_PSW_sim0300', visual=False, show_input_maps=False)
pcat_test_sim.procure_cirrus_realizations(5)
#if __name__ == '__main__':																			     

#	sim_idx = int(sys.argv[1])+int(sys.argv[2])
#	print('sim index is ', sim_idx)
##	if os.path.exists(started_idx_list_file):
#		started_sim_idxs = list(np.load(started_idx_list_file)['sim_idxs'])
#		while sim_idx in started_sim_idxs:
#		      sim_idx += 1
#		print('sim idx is now', sim_idx)
#		started_sim_idxs.append(sim_idx)
#		np.savez(started_idx_list_file, sim_idxs=started_sim_idxs)
#	else:
#		np.savez(started_idx_list_file, sim_idxs=[sim_idx])
		      

#	print('sim indeeeex is ', sim_idx)																   
#	pcat_test = pcat_test_suite(cluster_name='rxj1347', base_path=base_path, result_path=result_path, cblas=False, openblas=False)
#	pcat_test.iter_fourier_comps(dataname='new_gen3_sims', tail_name='rxj1347_PSW_sim0'+str(sim_idx), \
#							nsamps=[50, 100, 200, 500], final_nsamp=2000, visual=False, show_input_maps=False, timestr_list_file=timestr_list_file)



#        run_pcat(sim_idx=sim_idx_0+sim_idx)                                                                                                                                          
#        run_pcat_dust_and_sz_test(sim_idx=sim_idx_0+sim_idx, inject_dust=True, inject_sz_frac=1.0)                                                                                   




#pcat_test = pcat_test_suite(cluster_name='rxj1347', base_path=base_path, result_path=result_path, cblas=False, openblas=False)
#pcat_test.iter_fourier_comps(dataname='new_gen3_sims', tail_name='rxj1347_PSW_sim0300', \
#							nsamps=[10, 10, 10, 10], final_nsamp=30, visual=False, show_input_maps=False, timestr_list_file='test_11_10_20.npz')



# base_path = '/Users/luminatech/Documents/multiband_pcat/'
# result_path = '/Users/luminatech/Documents/multiband_pcat/spire_results/'

# initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.002, 'L':0.01}), 'planck': dict({'S':1.0, 'M':1.0, 'L':1.0})})

# t_filenames = [base_path+'Data/spire/rxj1347/rxj1347_PSW_nr.fits']

# template_names = ['sze']

# #cluster_name='rxj1347'
# cluster_name= 'as0592'


# n_fc_terms = 10

# init_fc = np.zeros(shape=(n_fc_terms, n_fc_terms, 4))

# # fmin_levels = [0.05, 0.01, 0.005]
# # nsamps = [50, 100, 500]

# fmin_levels = [0.05, 0.02, 0.01, 0.007]
# nsamps = [50, 100, 200, 500]


# median_fcs_iter = []

# template_filename=dict({'sze': t_filenames[0]})

# timestr = None

# for i, fmin in enumerate(fmin_levels):
# 	if i==0:
# 		print('initial fourier coefficients set to zero')
# 		median_fc = init_fc

# 	initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.000, 'L':0.014})})

# 	# ob = lion(band0=0, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=0,\
# 	# 	 cblas=True, openblas=False, visual=False, show_input_maps=False, float_templates=False, tail_name=cluster_name+'_PSW_nr_1_ext',\
# 	# 	  dataname='rxj1347_831', bias=[-0.004], load_state_timestr=timestr, max_nsrc=1000, auto_resize=True, trueminf=fmin, nregion=5, weighted_residual=True,\
# 	# 	   make_post_plots=False, nsamp=nsamps[i], use_mask=True, residual_samples=5, init_fourier_coeffs=median_fc, n_frames=3, float_fourier_comps=True, n_fourier_terms=n_fc_terms,\
# 	# 	    show_fc_temps=False, fc_sample_delay=0, fourier_comp_moveweight=200., alph=1.0, dfc_prob=1.0)

# 	ob = lion(band0=0, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=0,\
# 		 cblas=True, openblas=False, visual=True, show_input_maps=False, float_templates=False, tail_name='as0592_PSW_nr_1_ext',\
# 		  dataname='as0592_925', bias=[-0.004], load_state_timestr=timestr, max_nsrc=1000, auto_resize=True, trueminf=fmin, nregion=5, weighted_residual=True,\
# 		   make_post_plots=False, nsamp=nsamps[i], use_mask=True, residual_samples=5, init_fourier_coeffs=median_fc, n_frames=3, float_fourier_comps=True, n_fourier_terms=n_fc_terms,\
# 		    show_fc_temps=False, fc_sample_delay=0, fourier_comp_moveweight=200., alph=1.0, dfc_prob=1.0)

# 	# ob = lion(band0=0, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=0, \
# 	# 		 cblas=True, openblas=False, visual=True, show_input_maps=False, float_templates=False, tail_name='rxj1347_PSW_nr_1_ext',\
# 	# 		  dataname='rxj1347_831', bias=[-0.004], load_state_timestr=timestr,  max_nsrc=600, auto_resize=True, trueminf=fmin, nregion=5, weighted_residual=True,\
# 	# 		  nsamp=nsamps[i], use_mask=True, make_post_plots=False, residual_samples=20, init_fourier_coeffs=median_fc, n_frames=3, float_fourier_comps=True, n_fourier_terms=n_fc_terms, fc_sample_delay=0, fourier_comp_moveweight=40.)

# 	ob.main()
# 	_, filepath, _ = load_param_dict(ob.gdat.timestr)
# 	timestr = ob.gdat.timestr

# 	chain = np.load(filepath+'/chain.npz')
# 	median_fc = np.median(chain['fourier_coeffs'][-10:], axis=0)

# 	median_fcs_iter.append(median_fc)
# 	median_fc_temp = generate_template(median_fc, n_fc_terms, N=ob.gdat.imsz0[0], M=ob.gdat.imsz0[1])

# 	plt.figure()
# 	plt.imshow(median_fc_temp, cmap='Greys', interpolation=None, origin='lower')
# 	plt.colorbar()
# 	plt.savefig('figures/median_fc_temp_'+cluster_name+'_iter'+str(i)+'_10_28_20_nfcterms='+str(n_fc_terms)+'_deconvolved.png', bbox_inches='tight')
# 	# plt.savefig('median_fc_temp_iter'+str(i)+'_10_20_20_nfcterms='+str(n_fc_terms)+'_'+cluster_name+'_'+str(sim_idx)+'.png', bbox_inches='tight')
# 	# plt.savefig('median_fc_temp_iter'+str(i)+'_10_13_20_nfcterms='+str(n_fc_terms)+'_as0592.png', bbox_inches='tight')
# 	plt.close()

# # def run_pcat_dust_and_sz_test(sim_idx=200, inject_dust=False, show_input_maps=False, inject_sz_frac=1.0):


# 	# ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.6, bkg_sig_fac=5.0, bkg_sample_delay=10, temp_sample_delay=20, \
# 	# 		 cblas=True, openblas=False, visual=False, show_input_maps=show_input_maps, float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, tail_name='rxj1347_PSW_sim0'+str(sim_idx)+'_dust',\
# 	# 		  dataname='sim_w_dust', bias=None, max_nsrc=1500, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True,\
# 	# 		   make_post_plots=True, nsamp=50, delta_cp_bool=True, use_mask=True, residual_samples=100, template_filename=t_filenames, inject_dust=inject_dust, inject_sz_frac=inject_sz_frac)
# 	# ob.main()

# final_temp = generate_template(median_fc, n_fc_terms, N=ob.gdat.imsz0[0], M=ob.gdat.imsz0[1])

# np.savez('median_fc_iter'+'_10_28_20_nfcterms='+str(n_fc_terms)+'_'+cluster_name+'.npz', median_fcs_iter=median_fcs_iter)

# initial_template_amplitude_dicts = dict({'sze': dict({'S':0.00, 'M':0.000, 'L':0.014})})

# template_names = ['sze']

# # _, filepath, _ = load_param_dict('20201014-175849')

# chain = np.load(filepath+'/chain.npz')
# median_fc = np.median(chain['fourier_coeffs'][-10:], axis=0)
# template_filename=dict({'sze': t_filenames[0]})

# # ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=0,\
# # 	 cblas=True, openblas=False, visual=True, show_input_maps=False, float_templates=False, tail_name=cluster_name+'_PSW_nr_1_ext',\
# # 	  dataname='rxj1347_831', bias=[-0.004, -0.007, -0.008], load_state_timestr=timestr, max_nsrc=1000, auto_resize=True, trueminf=0.007, nregion=5, weighted_residual=True,\
# # 	   make_post_plots=False, nsamp=nsamps[i], use_mask=True, residual_samples=5, init_fourier_coeffs=median_fc, n_frames=3, float_fourier_comps=True, n_fourier_terms=n_fc_terms,\
# # 	    show_fc_temps=False, fc_sample_delay=0, fourier_comp_moveweight=0., alph=1.0, dfc_prob=1.0, fc_rel_amps=[1.0, 0.55, 0.25])

# fmin = 0.007

# flux_density_conversion_dict = dict({'S': 86.29e-4, 'M':16.65e-3, 'L':34.52e-3})

# dust_I_lams = [3e6, 1.6e6, 8e5] # in Jy/sr
# band_dict = dict({0:'S', 1:'M', 2:'L'})

# dust_rel_SED = [dust_I_lams[i]/dust_I_lams[0] for i in range(len(dust_I_lams))]
# fc_rel_amps = [dust_rel_SED[i]*flux_density_conversion_dict[band_dict[i]]/flux_density_conversion_dict[band_dict[0]] for i in range(len(dust_I_lams))]

# ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=0, temp_sample_delay=50,\
# 		 cblas=True, openblas=False, visual=True, show_input_maps=True, float_templates=False, tail_name='as0592_PSW_nr_1_ext',\
# 		  dataname='as0592_925', bias=[-0.004, -0.007, -0.008], delta_cp_bool=True, max_nsrc=1000, init_fourier_coeffs=median_fc, auto_resize=True, trueminf=fmin, nregion=5, weighted_residual=True,\
# 		   make_post_plots=False, nsamp=2000, use_mask=True, residual_samples=200, template_filename=None, float_fourier_comps=True, show_fc_temps=False, n_fourier_terms=n_fc_terms, fc_sample_delay=0, fourier_comp_moveweight=0., \
# 		   movestar_sample_delay=0, merge_split_sample_delay=0, birth_death_sample_delay=0, alph=1.0, dfc_prob=0.0, fc_rel_amps=fc_rel_amps)


# # ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=0, temp_sample_delay=50,\
# # 		 cblas=True, openblas=False, visual=True, show_input_maps=True, float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, tail_name='rxj1347_PSW_nr_1_ext',\
# # 		  dataname='rxj1347_831', bias=[-0.004, -0.007, -0.008], delta_cp_bool=True, max_nsrc=1000, init_fourier_coeffs=median_fc, auto_resize=True, trueminf=fmin, nregion=5, weighted_residual=True,\
# # 		   make_post_plots=False, nsamp=2000, use_mask=True, residual_samples=200, template_filename=None, float_fourier_comps=True, show_fc_temps=False, n_fourier_terms=n_fc_terms, fc_sample_delay=0, fourier_comp_moveweight=0., \
# # 		   movestar_sample_delay=0, merge_split_sample_delay=0, birth_death_sample_delay=0, alph=1.0, dfc_prob=0.0, fc_rel_amps=fc_rel_amps)

# ob.main()

# np.savez('median_fc_iter'+'_10_27_20_nfcterms='+str(n_fc_terms)+'_rxj1347.npz', median_fcs_iter=median_fcs_iter)

# for i in range(2):
# 	viz = True
# 	if i > 0:
# 		viz = False

# 	ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=0, temp_sample_delay=50,\
# 			 cblas=True, openblas=False, visual=viz, show_input_maps=False, float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, tail_name='rxj1347_PSW_nr_1_ext',\
# 			  dataname='rxj1347_831', bias=[-0.004, -0.007, -0.008], delta_cp_bool=True, max_nsrc=1000, init_fourier_coeffs=median_fc, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True,\
# 			   make_post_plots=True, nsamp=2000, use_mask=True, residual_samples=200, template_filename=None, float_fourier_comps=True, show_fc_temps=False, n_fourier_terms=n_fc_terms, fc_sample_delay=0, fourier_comp_moveweight=0., \
# 			   movestar_sample_delay=0, merge_split_sample_delay=0, birth_death_sample_delay=0, alph=1.0, dfc_prob=0.0, fc_rel_amps=[1.0, 0.5, 0.25])

# 	ob.main()

# ob = lion(band0=0, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=False, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=0,\
# 	 cblas=True, openblas=False, visual=True, show_input_maps=False, float_templates=False, tail_name=cluster_name+'_PSW_nr_1_ext',\
# 	  dataname='rxj1347_831', bias=[-0.004], load_state_timestr=timestr, max_nsrc=1000, auto_resize=True, trueminf=fmin, nregion=5, weighted_residual=True,\
# 	   make_post_plots=False, nsamp=nsamps[i], use_mask=True, residual_samples=20, init_fourier_coeffs=median_fc, n_frames=3, float_fourier_comps=True, n_fourier_terms=n_fc_terms,\
# 	    show_fc_temps=False, fc_sample_delay=0, fourier_comp_moveweight=40., alph=1.0)

# np.savez('median_fc_iter'+'_10_13_20_nfcterms='+str(n_fc_terms)+'_as0592.npz', median_fcs_iter=median_fcs_iter)


