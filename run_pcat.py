from pcat_spire import *

'''These lines below here are to instantiate the script without having to load an updated 
version of the lion module every time I make a change, but when Lion is wrapped within another pipeline
these should be moved out of the script and into the pipeline'''

base_path = '/Users/luminatech/Documents/multiband_pcat/'
result_path = '/Users/luminatech/Documents/multiband_pcat/spire_results/'

# def run_pcat(sim_idx=302, trueminf=0.005):
# 	ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=50,\
# 			 cblas=True, openblas=False, visual=False, float_templates=True, template_names=['sze'], template_amplitudes=[[0.0], [0.0], [0.0]], tail_name='rxj1347_PSW_sim0'+str(sim_idx),\
# 			  dataname='sims_for_richard', bias=[-0.003, -0.003, -0.003], max_nsrc=1200, auto_resize=True, trueminf=trueminf, nregion=5, weighted_residual=True,\
# 			   make_post_plots=True, nsamp=2000, residual_samples=300, template_filename=['Data/spire/rxj1347/rxj1347_PSW_nr_sze.fits'], \
# 			   inject_sz_frac= 0.5)
# 	ob.main()

initial_template_amplitude_dicts = dict({'sze': dict({'S':0.001, 'M':0.002, 'L':0.01}), 'dust': dict({'S':0.0, 'M':0.0, 'L':0.0})})
# initial_template_amplitude_dicts = dict({'sze': dict({'S':0.001, 'M':0.002, 'L':0.01})})
# initial_template_amplitude_dicts = dict({'dust': dict({'S':1.0, 'M':1.0, 'L':1.0})})

t_filenames = ['Data/spire/rxj1347/rxj1347_PSW_nr_sze.fits', 'Data/spire/rxj1347/dust_template_PSW.npz']
# t_filenames = ['Data/spire/rxj1347/rxj1347_PSW_nr_sze.fits']
# t_filenames = ['Data/spire/rxj1347/dust_template_PSW.npz']

template_names = ['sze', 'dust']
# template_names = ['sze']
# template_names = ['dust']

# ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=50,\
# 		 cblas=True, openblas=False, visual=False, float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, tail_name='rxj1347_PSW_nr',\
# 		  dataname='rxj1347', bias=[-0.006, -0.008, -0.01], max_nsrc=1200, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True,\
# 		   make_post_plots=True, nsamp=4000, residual_samples=300, template_filename=t_filenames, \
# 		   )
# ob.main()


# ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.75, bkg_sig_fac=5.0, bkg_sample_delay=50,\
# 		 cblas=True, openblas=False, visual=True, float_templates=True, template_names=['sze', 'dust'], init_template_amplitudes=[[0.0], [0.0], [0.0]], tail_name='rxj1347_PSW_nr',\
# 		  dataname='rxj1347', bias=[-0.003, -0.003, -0.003], max_nsrc=1200, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True,\
# 		   make_post_plots=True, nsamp=2000, residual_samples=300, template_filename=['Data/spire/rxj1347/rxj1347_PSW_nr_sze.fits', 'Data/spire/rxj1347/dust_template_PSW.npz'], \
# 		   )
# ob.main()


# # for i in range(306, 309):
# # 	run_pcat(sim_idx=i)
# # run_pcat(sim_idx=300)

# timestr_list_threeband_sz = ['20200607-010226', '20200607-010206', '20200607-010142', '20200605-172052']

# timestr_list_twoband_sz = ['20200607-130436', '20200607-130411', '20200607-130345', '20200605-172157']

# for timestring in timestr_list_threeband_sz:


# 	result_plots(timestr=timestring,cattype=None, burn_in_frac=0.75, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)

# # from multiprocessing import Pool

# def multithreaded_pcat(n_process=2):
# 	p = Pool(processes=n_process)
# 	p.map(run_pcat, [0. for x in range(n_process)])

# multithreaded_pcat()

# ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.8, bkg_sig_fac=5.0, bkg_sample_delay=0,\
# 			 cblas=False, openblas=True, visual=False, float_templates=True, template_names=['sze'], template_amplitudes=[[0.0], [0.0], [0.0]], tail_name='rxj1347_PSW_sim0302',\
# 			  dataname='new_sporc_sims', bias=[-0.003, -0.003, -0.003], max_nsrc=2000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True,\
# 			   make_post_plots=True, nsamp=50, residual_samples=10, template_filename=['Data/spire/rxj1347/rxj1347_PSW_nr_sze.fits'])



# ob = lion(band0=0, band1=1, band2=2, cblas=True, visual=True, dataname='a0370', tail_name='PSW_nr_1', mean_offsets=[0.0, 0.0, 0.0], auto_resize=False, x0=70, y0=70, width=100, height=100, trueminf=0.001, nregion=5, weighted_residual=False, make_post_plots=True, nsamp=50, residual_samples=10)

# real data, rxj1347, floating SZ templates
# ob = lion(band0=0, band1=1, band2=2, cblas=True, visual=False, float_templates=True, template_names=['sze'], template_amplitudes=[[0.0], [0.1], [0.1]], tail_name='PSW_nr', dataname='rxj1347', mean_offsets=[0., 0., 0.], bias=[0.0, 0.0, 0.0], max_nsrc=3000,x0=70, y0=70, width=100, height=100, auto_resize=False, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=100)
# ob = lion(band0=0, band1=1, band2=2, bolocam_mask=True, noise_thresholds=[0.002, 0.002, 0.003], float_background=True, burn_in_frac=0.8, bkg_sig_fac=5.0, bkg_sample_delay=0, cblas=True, visual=False, float_templates=True, template_names=['sze'], template_amplitudes=[[0.0], [0.003], [0.01]], tail_name='PSW_nr', dataname='rxj1347', bias=[-0.003, -0.005, -0.008], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=2000, residual_samples=400)
# ob = lion(band0=0, band1=1, band2=2, bolocam_mask=True, float_background=True, burn_in_frac=0.8, bkg_sig_fac=5.0, bkg_sample_delay=50, cblas=True, visual=False, float_templates=True, template_names=['sze'], template_amplitudes=[[0.0], [0.003], [0.003]], tail_name='PSW_nr', dataname='rxj1347', bias=[-0.003, -0.005, -0.008], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=3000, residual_samples=400)

# ob = lion(band0=0, band1=1, band2=2, float_background=True, bkg_sig_fac=20., bkg_sample_delay=10, cblas=True, visual=False, float_templates=False, tail_name='PSW_nr', dataname='rxj1347', bias=[-0.003, -0.005, -0.008], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=100, residual_samples=20)
# ob = lion(band0=0, float_background=True, bkg_sample_delay=20, cblas=True, visual=True, float_templates=False, tail_name='PSW_nr', dataname='rxj1347', bias=[-0.003], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=100, residual_samples=20)
# ob = lion(band0=0, band1=1, float_background=True, bkg_sample_delay=20, cblas=True, visual=True, float_templates=False, tail_name='PSW_nr', dataname='rxj1347', bias=[-0.003, -0.005], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=100, residual_samples=20)

# real data, rxj1347

# ob = lion(band0=1, cblas=True, visual=False, template_names=['sze'], template_amplitudes=[0.005], tail_name='PSW_nr', dataname='rxj1347', mean_offsets=[-0.003],x0=75, y0=75, width=100, height=100, max_nsrc=3000, auto_resize=False, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=100)
# ob = lion(band0=0, cblas=True, visual=True,tail_name='PSW_nr', dataname='rxj1347', mean_offsets=[-0.003], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=100)

# real data, abell 0068? 
# ob = lion(band0=0, cblas=True, visual=True, tail_name='PSW_6_8.2', dataname='a0068', x0=20, y0=20, width=200, height=200, bkg_sample_delay=20, bkg_sig_fac=20.0,float_background=True, mean_offsets=[0.], bias=[-0.005], max_nsrc=3000, auto_resize=False, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=500, residual_samples=100)


# ob = lion(band0=0, cblas=True, visual=False,verbtype=0, dataname='rxj1347', mean_offsets=[0.003, 0.006, 0.011], auto_resize=True, trueminf=0.002, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=200)

# simulated data, rxj1347
# ob = lion(band0=0, bkg_sample_delay=5, bkg_sig_fac=20.0, cblas=True, visual=True, verbtype=0, float_background=True, dataname='rxj1347', mean_offsets=[0.0], bias=[0.003], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=100)
# ob = lion(band0=0, band1=1, linear_flux = True, bkg_sample_delay=20, bkg_sig_fac=20.0, cblas=True, visual=False, verbtype=0, float_background=True, dataname='rxj1347', mean_offsets=[0.0, 0.0], bias=[0.003, 0.002], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=100)
# ob = lion(band0=0, band1=1, band2=2, linear_flux = True, bkg_sample_delay=20, bkg_sig_fac=20.0, cblas=True, visual=False, verbtype=0, float_background=True, dataname='rxj1347', mean_offsets=[0.0, 0.0, 0.0], bias=[0.003, 0.002, 0.007], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=100)
# ob.main()


# ob = lion(band0=0, band1=1, band2=2,tail_name='PSW_nr', bkg_sample_delay=50, cblas=True, visual=False, verbtype=0, float_background=True, dataname='rxj1347', mean_offsets=[0.0, 0.00, 0.00], bias=[0.000, 0.002, 0.007], max_nsrc=2000, auto_resize=True, trueminf=0.002, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=500, residual_samples=100)
# ob = lion(band0=0, bkg_sample_delay=50, tail_name='PSW_nr', bkg_sig_fac=20.0, cblas=True, visual=False, verbtype=0, float_background=True, dataname='rxj1347', mean_offsets=[0.0], bias=[-0.004], max_nsrc=3000, auto_resize=True, trueminf=0.003, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=100)

# ob = lion(band0=0, band1=1, band2=2, bkg_sample_delay=50, tail_name='PSW_nr', bkg_sig_fac=20.0, cblas=True, visual=False, verbtype=0, float_background=True, dataname='rxj1347', mean_offsets=[0.0, 0.0, 0.0], bias=[-0.004, -0.006, -0.007], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=3000, residual_samples=300)

# ob = lion(band0=0,bkg_sample_delay=50, bkg_sig_fac=20.0, cblas=True, visual=False, verbtype=0, float_background=True, dataname='rxj1347', mean_offsets=[0.0], bias=[0.005], max_nsrc=3000, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=100)

# ob = lion(band0=0, band1=1, band2=2, cblas=True, visual=False, verbtype=0, float_background=True, dataname='rxj1347', mean_offsets=[0.0, 0.00, 0.00], bias=[0.002, 0.005, 0.011], max_nsrc=2000, auto_resize=True, trueminf=0.002, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=200)

# ob = lion(band0=0, cblas=True, visual=False,verbtype=0, dataname='rxj1347', mean_offsets=[0.003, 0.006, 0.011], auto_resize=True, trueminf=0.002, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=200)


# simulated rxj1347
# ob = lion(band0=0, band1=1, cblas=True, visual=True,verbtype=0, tail_name='PSW_0', dataname='sides_cat',mean_offsets=[0.,0.], bias=[0.003, 0.006], auto_resize=True, trueminf=0.003, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=200)
# ob = lion(band0=0, cblas=True, visual=True,verbtype=0, tail_name='PSW_0', dataname='sides_cat',mean_offsets=[0.], bias=[0.001], auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=200)


# ob = lion(band0=0, band1=1, cblas=True, visual=False,verbtype=0, dataname='rxj1347',mean_offsets=[0.,0.], bias=[0.003, 0.006], auto_resize=True, trueminf=0.003, nregion=5, weighted_residual=True, make_post_plots=True, nsamp=1000, residual_samples=200)
# ob.main()

# ob = lion(band0=0, band1=1, band2=2, visual=False, openblas=True, cblas=False, auto_resize=True, make_post_plots=True, nsamp=100, residual_samples=100, weighted_residual=True)
# ob = lion(band0=0, openblas=True, visual=True, cblas=False, x0=50, y0=50, width=100, height=60, nregion=5, make_post_plots=True, nsamp=100, residual_samples=100, weighted_residual=True)


# result_plots(timestr='20200618-010116',cattype=None, burn_in_frac=0.75, boolplotsave=True, boolplotshow=False, plttype='png', gdat=None)

# ob_goodsn = lion(band0=0, mean_offset=0.005, cblas=True, visual=False, auto_resize=False, width=200, height=200, x0=150, y0=150, trueminf=0.015, nregion=5, dataname='GOODSN_image_SMAP', nsamp=5, residual_samples=1, max_nsrc=2500, make_post_plots=True)
# ob_goodsn = lion(band0=0, band1=1, cblas=True, visual=True, auto_resize=True, trueminf=0.001, nregion=5, dataname='GOODSN_image_SMAP', nsamp=200, residual_samples=50, max_nsrc=2000, make_post_plots=True)
# ob_goodsn.main()