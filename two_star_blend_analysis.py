import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

offsets = np.array([0.5,0.6, 0.75, 1.0, 1.2, 1.5], dtype=np.float32)
#flux_ratios = np.array([1.0], dtype=np.float32)
#r_fluxes = np.array([1000.0], dtype=np.float32)

flux_ratios = np.array([1.0, 2.0, 5.0], dtype=np.float32)
r_fluxes = np.array([500.0, 1000.0], dtype=np.float32)
#r_fluxes = np.array([250.0], dtype=np.float32)
imsz = 30
nsamp = 100
tol = 2
pdf_or_png = 'png'
cases = ['r+i+g', 'r', 'rx3']
ncases = len(cases)

error_types = ['Position Error (pixels)', 'Fractional Flux Error']
figure_labels = ['position_error_src', 'flux_error_src']
error_indices = [5, 7]
y_lim_types = [0.8, 1.2]

r_i_colors = [0.3, 0.1]
r_g_colors = [-0.5, -0.1]
src_colors = [r_i_colors, r_g_colors]



flux_fac = [1+10**(0.4*src_colors[0][0])+10**(0.4*src_colors[1][0]), 1+10**(0.4*src_colors[0][1])+10**(0.4*src_colors[1][1])]


mock_test_name = 'mock_2star_'+str(imsz)
if sys.platform=='darwin':
    directory_path = '/Users/richardfeder/Documents/multiband_pcat/pcat-lion-master'
    result_path = '/Users/richardfeder/Documents/multiband_pcat/two_star_blend'
elif sys.platform=='linux2':
    directory_path = '/n/fink1/rfeder/mpcat/multiband_pcat'
    result_path = '/n/home07/rfederstaehle/figures/mock_2star_'+str(imsz)
    print 'result_path:', result_path

if not os.path.isdir(result_path+'/'+mock_test_name):
	os.makedirs(result_path+'/'+mock_test_name)
	result_path = result_path+'/'+mock_test_name

def find_offsets_errs(f, rat, plist, offsets, num, case):
	sublist = [p for p in plist if p[1]==f and p[2]==rat and p[3]==case]
	offs, mean_errs, std_errs = [[] for x in xrange(3)]
	for offset in offsets:
		entry = np.array([p for p in sublist if p[0]==offset])
		if len(entry)>0:
			offs.append(offset)
			mean_errs.append(np.average(entry[:,num]))
			std_errs.append(np.std(entry[:,num]))
		else:
			offs.append(np.nan)
			mean_errs.append(np.nan)
			std_errs.append(0.)
	return offs, mean_errs, std_errs


def find_min_ds(xs, ys, fs, source, case, flux_ratios, r_fluxes, offsets, a, b, c, source_no):
	abs_nonz_dx = np.abs(xs-source[0])
	abs_nonz_dy = np.abs(ys-source[1])
	ds = np.square(abs_nonz_dx)+np.square(abs_nonz_dy)
	min_ds_arg = np.argmin(ds)
	minds = np.sqrt(ds[min_ds_arg]) # if ds within 2 offset separations of source, associate it, otherwise don't
	fminds = fs[min_ds_arg]
	if minds < 2*offsets[a]:
		if case==2:
			if source_no == 1: # don't use flux_ratio for this one
				frac_f = np.abs(fminds-flux_fac[0]*r_fluxes[b])/(flux_fac[0]*r_fluxes[b])
			elif source_no ==2: # use flux ratio to get true flux of brighter source here
				frac_f = np.abs(fminds-flux_fac[1]*r_fluxes[b]*flux_ratios[c])/(flux_fac[1]*r_fluxes[b]*flux_ratios[c])
		else:
			if source_no == 1:
				frac_f = np.abs(fminds-r_fluxes[b])/r_fluxes[b]
			elif source_no==2:
				frac_f = np.abs(fminds-r_fluxes[b]*flux_ratios[c])/(r_fluxes[b]*flux_ratios[c])

		return minds, frac_f, min_ds_arg
	else:
		return [0, 0, 0]

def load_arrays(a, b, c, nrealization=0):
	dataname = mock_test_name+'-' + str(offsets[a])+'-'+str(r_fluxes[b])+'-'+str(flux_ratios[c])

	chain_types = ['r+i+g', 'r', 'rx3']

	all_x, all_y, all_f, ns = [[] for x in xrange(4)]

	for chain in chain_types:
		if chain == 'rx3':
			dataname = mock_test_name+'-' + str(offsets[a])+'-'+str(3.*r_fluxes[b])+'-'+str(flux_ratios[c])
		if nrealization > 0:
			p = np.load(directory_path+'/Data/'+mock_test_name+'/'+dataname+'/results/'+chain+'-'+str(nrealization)+'.npz')
		else:
			p = np.load(directory_path+'/Data/'+mock_test_name+'/'+dataname+'/results/'+chain+'.npz')

		all_x.append(p['x'][-nsamp:])
		all_y.append(p['y'][-nsamp:])
		all_f.append(p['f'][0,-nsamp:])
		ns.append(p['n'][-nsamp:])

	return all_x, all_y, all_f, ns


def fp_error(imsz, nsam, a, b, c, nrealization, case, all_x, all_y, all_f, ns, src1, src2):

	dss1, dss2, dfs1, dfs2 = [[] for x in xrange(4)]

	for s in xrange(len(all_x[0])):

		# in both cases, we first associate to the brighter of the two sources (this will be source 2 by design), 
		# given a specific criterion, then we associate to source 1
		
		if ns[case][s]>1: # two sources or more in given sample

			nonz_x = all_x[case][s][np.nonzero(all_x[case][s])]
			nonz_y = all_y[case][s][np.nonzero(all_x[case][s])]
			nonz_f = all_f[case][s][np.nonzero(all_x[case][s])]
			minds2, frac_f2, min_ds_arg2 = find_min_ds(nonz_x, nonz_y, nonz_f, src2, case, flux_ratios, r_fluxes, offsets, a, b, c, 2)
			
			# if a source is associated, then log its position/flux errors and remove it from sample
			if minds2 > 0:
				dss2.append(minds2)
				dfs2.append(frac_f2)
				nonz_x = np.delete(nonz_x, min_ds_arg2) #remove first sample and repeat
				nonz_y = np.delete(nonz_y, min_ds_arg2)
				nonz_f = np.delete(nonz_f, min_ds_arg2)

			minds1, frac_f1, min_ds_arg1 = find_min_ds(nonz_x, nonz_y, nonz_f, src1, case, flux_ratios, r_fluxes, offsets, a, b, c, 1)
			if minds1 > 0:
				dss1.append(minds1)
			if frac_f1 > 0:
				dfs1.append(frac_f1)

		elif ns[case][s]==1: #if only one source in the sample
			nonz_x = all_x[case][s][0]
			nonz_y = all_y[case][s][0]
			nonz_f = all_f[case][s][0]
			ds2 = np.sqrt(np.square(nonz_x-src2[0])+np.square(nonz_y-src2[1]))

			if case==2:
				frac_f = np.abs(nonz_f-3*r_fluxes[b]*flux_ratios[c])/(3*r_fluxes[b]*flux_ratios[c])
			else:
				frac_f = np.abs(nonz_f-r_fluxes[b]*flux_ratios[c])/(r_fluxes[b]*flux_ratios[c])
			
			if flux_ratios[c] > 1: #give it directly to brighter source
				dss2.append(ds2)
				dfs2.append(frac_f)
			else:
				ds1 = np.sqrt(np.square(nonz_x-src1[0])+np.square(nonz_y-src1[1]))
				weights = [1/ds1**2, 1/ds2**2]
				weights /= np.sum(weights)
				choose_source = np.random.choice(weights.size, p=weights)
				if choose_source==0:
					dss1.append(ds1)
					dfs1.append(frac_f)
				else:
					dss2.append(ds2)
					dfs2.append(frac_f)

	return dss1, dss2, dfs1, dfs2


def flux_position_errors(imsz, nsam, nrealization=0):
	pos_error_list = [] #offset, r_flux, flux_ratio, mean_flux_err, mean_flux_err1, mean_flux_err3
	for a in xrange(len(offsets)):
		for b in xrange(len(r_fluxes)):
			for c in xrange(len(flux_ratios)):

				all_x, all_y, all_f, ns = load_arrays(a,b,c,nrealization)

				src1 = [int(imsz/2), int(imsz/2)] # true positions of source 1
				src2 = [int(imsz/2)+offsets[a], int(imsz/2)] # true positions of source 2
				
				pos_errs1, pos_errs2, ferrs1, ferrs2 = [[] for x in xrange(4)]
				
				for case in xrange(len(cases)):

					dss1, dss2, dfs1, dfs2 = fp_error(imsz, nsam,a,b,c, nrealization, case, all_x, all_y, all_f, ns, src1, src2)

					pos_error_list.append([offsets[a], r_fluxes[b], flux_ratios[c], case, nrealization, np.mean(dss1), np.mean(dss2), np.mean(dfs1), np.mean(dfs2)])

	return pos_error_list 

def get_prevalence(nrealization=0):
	nstar_vals = []
	for a in xrange(len(offsets)):
		xmin = (imsz/2)-tol
		ymin = (imsz/2)-tol
		xmax = (imsz/2)+offsets[a]+tol
		ymax = (imsz/2)+tol
		for b in xrange(len(r_fluxes)):
			for c in xrange(len(flux_ratios)):
				all_x, all_y, all_f, ns = load_arrays(a,b,c, nrealization)
				for case in xrange(ncases):
					nstar = []
					for n in xrange(len(ns[case])):
						# this line is tricky, since one source may be in the region and the second may not be
						if np.any(all_x[case][n]>xmin) and np.any(all_x[case][n])<xmax and np.any(all_y[case][n]>ymin) and np.any(all_y[case][n]<ymax):
							nstar.append(ns[case][n])

					onestar_prevalence = float(len([x for x in nstar if x==1]))/float(len(nstar))
					twostar_prevalence = float(len([x for x in nstar if x==2]))/float(len(nstar))
					morestar_prevalence = float(len([x for x in nstar if x>2]))/float(len(nstar))
					nstar_vals.append([offsets[a], r_fluxes[b], flux_ratios[c], case, nrealization, twostar_prevalence, onestar_prevalence, morestar_prevalence])

	return nstar_vals   


def make_error_plots(error_types, full_plist, error_indices, source, realizations):
    marker_colors = ['g', 'r', 'b']
    for err_type in xrange(len(error_types)):
        c=1
        plt.figure(figsize=(10,10), dpi=200)
        for flux in r_fluxes:
            for ratio in flux_ratios:
                plt.subplot(3,3,c)
                plt.title('$f_1$ = '+str(flux)+', $f_2/f_1$ = '+str(ratio))
                for case in xrange(ncases):
                    off, mean_err, std_err = find_offsets_errs(flux, ratio, full_plist, offsets, error_indices[err_type]+source, case)
                    plt.plot(off, mean_err, color=marker_colors[case])
                    if len(realizations) > 1:
                        plt.errorbar(off, mean_err, yerr=std_err, label=cases[case], fmt='o', color=marker_colors[case], capsize=5, markeredgewidth=2 )
                    else:
                        plt.scatter(off, mean_err, label=cases[case])
				
                    if c==3:
                        plt.legend(loc=1)
                    if c%3==1:
                        plt.ylabel(error_types[err_type])
                    if c > 6:
                        plt.xlabel('Source Separation (pixels)')
				
                    plt.ylim(0, y_lim_types[err_type])
                c+=1
            plt.tight_layout()
            plt.legend()
        if sys.platform =='linux2':
            plt.savefig(result_path+'/'+mock_test_name+'_'+figure_labels[err_type]+str(source+1)+'_py.'+pdf_or_png, bbox_inches='tight')
        else:
            plt.savefig(directory_path+'/Data/'+mock_test_name+'/'+figure_labels[err_type]+str(source+1)+'_py.'+pdf_or_png, bbox_inches='tight')

def get_plist(realizations):
	full_plist = []
	for realization in realizations:
		plist = flux_position_errors(imsz, nsamp, realization)
		full_plist.extend(plist)
	return full_plist

def get_full_nstar(realizations):
	full_nstar_vals = []
	for realization in realizations:
		print 'realization:', realization
		nstar_vals = get_prevalence(realization)
		full_nstar_vals.extend(nstar_vals)
	return full_nstar_vals

def plot_prevalences(full_nstar_vals, realizations):
    marker_colors  = ['g', 'r', 'b']
    for flux in r_fluxes:
        c = 1
        plt.figure(figsize=(10,10), dpi=200)
        for ratio in flux_ratios:
            for numcase in xrange(3):
                plt.subplot(3,3,c)
                plt.title('($f_1$ = '+str(flux)+', $f_2/f_1$ = ' +str(ratio)+')')
                plt.ylim(-0.1,1.1)
                print 'ncases:', ncases
                for case in xrange(ncases):
                    off, prev, err = find_offsets_errs(flux, ratio, full_nstar_vals, offsets, 5+numcase, case) 
                    plt.plot(off, prev, color=marker_colors[case])
                    if len(realizations) > 1:
                        plt.errorbar(off, prev, yerr=err, label=cases[case], fmt='o', color=marker_colors[case], capsize=5, markeredgewidth=2)
                    else:
                        plt.scatter(off, prev, label=cases[case])
                    if c==3:
                        plt.legend(loc=1)
                    if c%3 ==1:
                        plt.ylabel('2-Source Prevalence')
                    elif c%3 ==2:
                        plt.ylabel('1-Source Prevalence')
                        plt.tick_params(labelleft=False)
                    elif c%3 == 0:
                        plt.ylabel('> 2-Source Prevalence')
                        plt.tick_params(labelleft=False)
                    if c > 6:
                        plt.xlabel('Source Separation (pixels)')
                    else:
                        plt.tick_params(labelbottom=False)
                c += 1
            plt.tight_layout()
            plt.legend()
            if sys.platform =='linux2':
                plt.savefig(result_path+'/'+mock_test_name+'_prevalence_panels_'+str(flux)+'_py.'+pdf_or_png, bbox_inches='tight')
            else:
                plt.savefig(directory_path+'/Data/'+mock_test_name+'/prevalence_panels_'+str(flux)+'_py.'+pdf_or_png, bbox_inches='tight')



# ------------------ POSITION AND FLUX ERRORS, PREVALENCE PLOTS  ----------------------

def all_plots(realizations):

	full_plist = get_plist(realizations)

	for source in xrange(2): # calculate errors for each source
		print 'Source:', source
		make_error_plots(error_types, full_plist, error_indices, source, realizations)

	full_nstar_vals = get_full_nstar(realizations)
	plot_prevalences(full_nstar_vals, realizations)



num_realizations = 3
spec_realizations = [1, 2, 3]
individual = 0

if individual:
	for x in xrange(1, num_realizations+1):
		realizations = [x]

		all_plots(realizations)

else:
	all_plots(spec_realizations)





