from pcat_spire import *


base_path = '/home/mbzsps/multiband_pcat/'
result_path = '/home/mbzsps/multiband_pcat/spire_results/'


template_names = ['sze', 'planck']
t_filenames = ['Data/spire/rxj1347/rxj1347_PSW_nr_sze.fits', 'Data/spire/rxj1347/dust_template_PSW.npz']

# template_names = ['planck']                                                                                                                  
# t_filenames = ['Data/spire/rxj1347/dust_template_PSW.npz']                                                                                   
                                                                                                                                                
initial_template_amplitude_dicts = dict({'dust': dict({'S':1.0, 'M':1.0, 'L':1.0}), 'planck': dict({'S':1.0, 'M':1.0, 'L':1.0}),'sze': dict({'S':0.0, 'M':0.001, 'L':0.02})})



def run_pcat_dust_and_sz_test(sim_idx=200, inject_dust=False, show_input_maps=False, inject_sz_frac=1.0):


        ob = lion(band0=0, band1=1, band2=2, base_path=base_path, result_path=result_path, round_up_or_down='down', bolocam_mask=True, float_background=True, burn_in_frac=0.6, bkg_sig_fac=5.0, bkg_sample_delay=10, temp_sample_delay=40, cblas=False, openblas=False,verbtype=0, visual=False, show_input_maps=show_input_maps, float_templates=True, template_names=template_names, init_template_amplitude_dicts=initial_template_amplitude_dicts, tail_name='rxj1347_PSW_sim0'+str(sim_idx)+'_dust',\
                  dataname='sim_w_dust', bias=None, max_nsrc=600, auto_resize=True, trueminf=0.005, nregion=5, weighted_residual=True,make_post_plots=True, nsamp=2000, delta_cp_bool=True, use_mask=True, residual_samples=100, template_filename=t_filenames, inject_dust=inject_dust, inject_sz_frac=inject_sz_frac, timestr_list_file='rxj1347_mock_test_8_26_20_100sims.npz')
        ob.main()

if __name__ == '__main__':

        sim_idx_0 = int(sys.argv[1])
        sim_idx = int(sys.argv[2])
 
        print('sim indeeeex is ', sim_idx_0+sim_idx)
#        run_pcat(sim_idx=sim_idx_0+sim_idx)
        run_pcat_dust_and_sz_test(sim_idx=sim_idx_0+sim_idx, inject_dust=True, inject_sz_frac=1.0)






