import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from helpers import *
from image_eval import image_model_eval
import h5py
import sys
import os

if sys.platform=='darwin':
    result_path = '/Users/richardfeder/Documents/multiband_pcat/pcat-lion-master/pcat-lion-results/'
    data_path = '/Users/richardfeder/Documents/multiband_pcat/pcat-lion-master/Data'
elif sys.platform=='linux2':
    result_path = '/n/home07/rfederstaehle/figures/'
    data_path = '/n/fink1/rfeder/mpcat/multiband_pcat/Data'
    #chain_path = '/n/fink1/rfeder/mpcat/multiband_pcat/pcat-lion-results/'
    chain_path = '/n/home07/rfederstaehle/pcat-lion-results/'
else:
    base_path = raw_input('Operating system not detected, please enter base_path directory (eg. /Users/.../pcat-lion-master):')
    if not os.path.isdir(base_path):
        raise OSError('Directory chosen does not exist. Please try again.')


np.seterr(divide='ignore', invalid='ignore')

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

run = '008151'
camcol = '4'
field = '0063'
mag_bins = np.linspace(15, 23, 15)

run_cam_field = run+'-'+camcol+'-'+field
bands = []

len_sys = len(sys.argv)
dataname = str(sys.argv[1])
result_dir_name = str(sys.argv[2])
datatype = str(sys.argv[3])
for b in xrange(4,len_sys):
    bands.append(sys.argv[b])
    

ref_cat_path = data_path+'/'+dataname+'/truth/'+dataname+'-tru.txt'
result_path += result_dir_name
chain_path += result_dir_name
print 'result_path: ', result_path
if not os.path.isdir(result_path):
    os.makedirs(result_path)
    os.makedirs(result_path+'/color_histogram')
    os.makedirs(result_path+'/frames')
    os.makedirs(result_path+'/residuals')

def look_at_chi2(filepath, figpath, nstart, nstop=None):
    chain = np.load(filepath)
    diff2s = chain['diff2s']
    if nstop is None:
        nstop =diff2s.shape[0]

    diff2 = diff2s[nstart:nstop,:].flatten()
    x =np.linspace(nstart*1000, nstop*1000, 1000*(nstop-nstart))

    plt.figure(figsize=(10,10))
    plt.plot(x,diff2, label='All Samples')
    plt.legend()
    plt.savefig(figpath)


def result_plots(result_path, ref_cat_path, \
                    hubble_cat_path=None, \
                    chain_datatype='npz', \
                    mock2_type=None, \
                    datatype='real', \
                    boolplotsave=1, \
                    boolplotshow=0, \
                    plttype='pdf', \
                    bright_n=50, \
                    burn_in_frac=0.3, \
                    bands=['r']):

    if datatype == 'mock':
        label = 'Mock Truth'
    else:
        label = datatype
    

    if chain_datatype.lower()=='npz':
        chain = np.load(chain_path+'/chain.npz')
    elif chain_datatype.lower()=='hdf5':
        chain = h5py.File(chain_path+'/chain.hdf5', 'r')
    else:
        raise IOError('Could not read in data type, please use .npz or .hdf5 files.')

    nsrcs = chain['n']
    xsrcs = chain['x']
    ysrcs = chain['y']
    fsrcs = chain['f']
    colorsrcs = chain['colors']
    chi2 = chain['chi2']
    timestats = chain['times'] # should be an array of shape (nsamp, 4)
    accept_stats = chain['accept']
    nmgy_per_count = chain['nmgy']
    epsilons = chain['eps']
    diff2s = chain['diff2s']
    #cprior_vals = np.array([[0,0],[0,0]])
    cprior_vals = chain['cprior_vals']
    nsamp = len(nsrcs)
    burn_in = int(nsamp*burn_in_frac)
    nbands = len(bands)
    ref_cat_dict = dict({"r":0, "i":1,"g":2, "z":2})
    ref_cat = np.loadtxt(ref_cat_path)
    ref_x = ref_cat[:,0]
    ref_y = ref_cat[:,1]
    ref_f = ref_cat[:,2:]
    ref_mags = []
    for b in xrange(nbands):
        print ref_cat_dict[bands[b]], ' index'
        ref_mag = adu_to_magnitude(ref_f[:,ref_cat_dict[bands[b]]], nmgy_per_count[b])
        ref_mags.append(ref_mag)

    ref_colors = []
    for b in xrange(nbands-1):
        ref_color = ref_mags[0]-ref_mags[b+1]
        ref_colors.append(ref_color)

        

    # ------------------- ABSOLUTE ASTROMETRIC OFFSET -------------

    for b in xrange(nbands-1):
        plt.figure()
        plt.hist(epsilons[-nsamp:,b,0],histtype='step', label='$\epsilon_x$')
        plt.hist(epsilons[-nsamp:,b,1],histtype='step', label='$\epsilon_y$')
        plt.legend()
        plt.xlabel('Absolute Astrometry Offset (pixels)')
        if boolplotsave:
            plt.savefig(result_path+'/posterior_histogram_epsilon.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()
    # ------------------- SOURCE NUMBER ---------------------------

    plt.figure()
    plt.title('Posterior Source Number Histogram')
    plt.hist(nsrcs[burn_in:], histtype='step', label='Posterior', color='b', bins=15)
    plt.axvline(np.median(nsrcs[burn_in:]), label='Median=' + str(np.median(nsrcs[burn_in:])), color='b', linestyle='dashed')
    plt.xlabel('nstar')
    plt.legend()
    if boolplotsave:
        plt.savefig(result_path +'/posterior_histogram_nstar.'+plttype, bbox_inches='tight')
    if boolplotshow:
        plt.show()
    plt.close()

    # -------------------- CHI2 ------------------------------------


    sample_number = list(xrange(nsamp-burn_in))
    full_sample = xrange(nsamp)
    plt.figure()
    plt.title('Chi-Squared Distribution over Catalog Samples')
    for b in xrange(nbands):
        plt.plot(sample_number, chi2[burn_in:,b], label=bands[b])
        plt.axhline(np.min(chi2[burn_in:,b]), linestyle='dashed', alpha=0.5, label=str(np.min(chi2[burn_in:,b]))+' (' + str(bands[b]) + ')')
    plt.xlabel('Sample')
    plt.ylabel('Chi2')
    plt.legend()
    if boolplotsave:
        plt.savefig(result_path + '/chi2_sample.'+plttype, bbox_inches='tight')
    if boolplotshow:
        plt.show()
    plt.close()

    # ---------------------------- COMPUTATIONAL RESOURCES --------------------------------

    time_array = np.zeros(4, dtype=np.float32)
    labels = ['Proposal', 'Likelihood', 'Implement', 'asTrans']
    for samp in xrange(nsamp):
        time_array += np.array([timestats[samp][2][0], timestats[samp][3][0]-timestats[samp][5][0], timestats[samp][4][0], timestats[samp][5][0]])
    plt.figure()
    plt.title('Computational Resources')
    plt.pie(time_array, labels=labels, autopct='%1.1f%%', shadow=True)
    if boolplotsave:
        plt.savefig(result_path+ '/time_resource_statistics.'+plttype, bbox_inches='tight')
    if boolplotshow:
        plt.show()
    plt.close()


    astrans_times = np.array(timestats)[:,5,0]

    plt.figure()
    plt.title('Time Histogram for asTrans Transformations')
    plt.hist(astrans_times, bins=20, histtype='step')
    plt.xlabel('Time (s)')
    plt.axvline(np.median(astrans_times), linestyle='dashed', color='r', label='Median: ' + str(np.median(astrans_times)))
    plt.legend()
    if boolplotsave:
        plt.savefig(result_path+ '/asTrans_time_resources.'+plttype, bbox_inches='tight')
    if boolplotshow:
        plt.show()
    plt.close()

    # ------------------------------ ACCEPTANCE FRACTION -----------------------------------------

    proposal_types = ['All', 'Move', 'Birth/Death', 'Merge/Split', 'Abs. Astrom']
    plt.figure()
    plt.title('Proposal Acceptance Fractions')
    for x in xrange(len(proposal_types)):
        if not np.isnan(accept_stats[0,x]):
            plt.plot(full_sample, accept_stats[:,x], label=proposal_types[x])
    plt.legend()
    plt.xlabel('Sample Number')
    plt.ylabel('Acceptance Fraction')
    if boolplotsave:
        plt.savefig(result_path+'/acceptance_fraction.'+plttype, bbox_inches='tight')
    if boolplotshow:
        plt.show()
    plt.close()


    # -------------------------------- ITERATE OVER BANDS -------------------------------------
    mag_bins = np.linspace(15, 23, 15)

    for b in xrange(nbands):

        plt.title('Posterior Magnitude Distribution - ' + str(bands[b]))
        (n, bins, patches) = plt.hist(ref_mags[b], histtype='step', label=label, color='g', bins=mag_bins)
        post_hist = []
        for samp in xrange(burn_in, nsamp):
            hist = np.histogram([adu_to_magnitude(x, nmgy_per_count[b]) for x in fsrcs[b][samp] if x>0], bins=bins)
            post_hist.append(hist[0]+0.01)
        medians = np.median(np.array(post_hist), axis=0)
        stds = np.std(np.array(post_hist), axis=0)
        bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
        plt.errorbar(bincentres, medians, yerr=stds, fmt='o', label='Posterior')
        plt.legend()
        plt.yscale('log', nonposy='clip')
        plt.ylim(0.05, 1000)
        plt.xlabel('Magnitude - ' + str(bands[b]))
        if boolplotsave:
            plt.savefig(result_path+'/posterior_flux_histogram_'+str(bands[b])+'.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()


    color_post_bins = np.linspace(-1.5, 1.5, 30)

    for b in xrange(nbands-1):

        nmpc = [nmgy_per_count[0], nmgy_per_count[b+1]]
        post_hist, bright_hist = [], []

        for samp in xrange(burn_in, samp):
            mask = fsrcs[0,samp,:] > 0
            n_bright = min(50, int(len(fsrcs[0,samp])/3))

            colors = colorsrcs[b][samp]
            brightest_idx = np.argpartition(fsrcs[0,samp][mask], n_bright)[-n_bright:]
            
            bright_h = np.histogram(colors[brightest_idx], bins=color_post_bins)
            hist = np.histogram(colors, bins=color_post_bins)
            post_hist.append(hist[0]+0.01)
            bright_hist.append(bright_h[0]+0.01)

        medians = np.median(np.array(post_hist), axis=0)
        medians /= (np.sum(medians)*(color_post_bins[1]-color_post_bins[0]))  
        medians_bright = np.median(np.array(bright_hist), axis=0)
        medians_bright /= (np.sum(medians_bright)*(color_post_bins[1]-color_post_bins[0]))  
        bincentres = [(color_post_bins[i]+color_post_bins[i+1])/2. for i in range(len(color_post_bins)-1)]

        cp_mu = cprior_vals[0,b]
        cp_sig = cprior_vals[1,b]
        cp_vals = gaussian(np.array(bincentres),cp_mu, cp_sig) 
        #print 'cp_sig:', cp_sig
        #print 'cp_mu:', cp_mu
        #print 'cp_vals:', cp_vals
        plt.figure()
        plt.title('Normalized Posterior Color Distribution')
        plt.step(bincentres, medians, where='mid', color='b', label='Posterior', alpha=0.5)
        plt.step(bincentres, medians_bright, where='mid', color='k', label='Brightest Third', alpha=0.5)
        plt.hist(ref_colors[b], color='g', bins=color_post_bins, label='DAOPHOT', histtype='step', normed=1, alpha=0.5)
        plt.plot(bincentres, cp_vals, color='r', label='Prior', alpha=0.5) #plot color prior as well
        plt.legend()
        plt.xlabel(str(bands[0]) + ' - ' + str(bands[b+1]))
        plt.legend()
        if boolplotsave:
            plt.savefig(result_path+'/posterior_histogram_'+str(bands[0])+'_'+str(bands[b+1])+'_color.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()

        plt.close()


# can now create a function that makes multiband sample frames but retroactively, based off of the chain samples and standard model evaluation. 

def multiband_retro_frames(result_path, ref_cat_path, data_path,\
                        hubble_cat_path=None, \
                        chain_datatype='npz', \
                        nframes=10, \
                        mock2_type=None, \
                        boolplotsave=1, \
                        boolplotshow=0, \
                        frame_number_list=None, \
                        datatype='real', \
                        plttype='pdf', \
                        bright_n=50, \
                        imdim=100,  \
                        bands=['r']):

    
    if datatype=='mock':
        labldata = 'Mock Truth'
    else:
        labldata = datatype

    sizefac = 10.*136

    if chain_datatype.lower()=='npz':
        chain = np.load(chain_path+'/chain.npz')
    elif chain_datatype.lower()=='hdf5':
        chain = h5py.File(chain_path+'/chain.hdf5', 'r')
    else:
        raise IOError('Could not read in data type, please use .npz or .hdf5 files.')


    nsrcs = chain['n']
    xsrcs = np.array(chain['x'], dtype=np.float32)
    ysrcs = np.array(chain['y'], dtype=np.float32)
    fsrcs = chain['f']
    colorsrcs = chain['colors']
    chi2 = chain['chi2']
    timestats = chain['times'] # should be an array of shape (nsamp, 5)
    acceptfracs = chain['accept']
    nmgy_per_count = chain['nmgy']
    bkgs = chain['back']
    pixel_transfer_mats = chain['pixel_transfer_mats']
    print 'backgrounds:', bkgs
    nsamp = len(nsrcs)
    nbands = len(bands)

    ref_cat = np.loadtxt(ref_cat_path)
    ref_x = ref_cat[:,0]
    ref_y = ref_cat[:,1]
    ref_f = ref_cat[:,2:]
    ref_mags = []
    for b in xrange(nbands):
        ref_mag = adu_to_magnitude(ref_f[:,b], nmgy_per_count[b])
        ref_mags.append(ref_mag)

    ref_colors = []
    for b in xrange(nbands-1):
        ref_color = ref_mags[b]-ref_mags[b+1]
        ref_colors.append(ref_color)


    hpath2 = '/n/fink1/rfeder/mpcat/multiband_pcat/Data/idR-002583-2-0136/hubble_catalog_2583-2-0136_astrans.txt'
    # use transform q to get hubble coordinates in other bands, or mean_dpos
    if hubble_cat_path is not None:
        hubble_coords, hf, hmask, posmask, hx, hy = get_hubble(hubble_cat_path, hpath_2=hpath2)
        #hubble_coords = np.loadtxt(hubble_cat_path)
        #hx0 = hubble_coords[:,0]-310
        #hy0 = hubble_coords[:,1]-630
        #h_rmag = hubble_coords[:,2]
    #hx1, hy1 = transform_q(h_coords[0][posmask], h_coords[1][posmask], pixel_transfer_mats[first_frame_band_no-

    psf_basic_path = data_path+'/'+dataname+'/psfs/'+dataname+'-psf.txt'
    cts_basic_path = data_path+'/'+dataname+'/cts/'+dataname+'-cts.txt'

    gains = dict()
    biases = dict()

    #if datatype != 'mock':
    #    frame_basic_path = data_path+'/'+dataname+'/frames/frame--'+run_cam_field+'.fits'
    #    gains = dict({"r":4.61999, "i":4.38999, "g":4.2, "z":5.67999})
    #    biases = dict({"r":1044., "i":1177., "g":1113., "z":1060.})
        
    
    for band in bands:
        pix_path = data_path+'/'+dataname+'/pixs/'+dataname+'-pix'+band+'.txt'
        g = open(pix_path)
        a = g.readline().split()
        bias, gain = [np.float32(i) for i in np.float32(g.readline().split())]
        gains[band] = gain
        biases[band] = bias

    #print 'biases:', biases
    #print 'gains:', gains
    #print 'nmgy_per_count:', nmgy_per_count


    color_bins = np.linspace(-4, 4, 50)

    #for band in bands:
        #if datatype=='mock':
            #pix_path = data_path+'/'+dataname+'/pixs/'+dataname+'-pix'+band+'.txt'
            #g = open(pix_path)
            #a = g.readline().split()
            #np.float32(g.readline().split())
            
            #frame_path = frame_basic_path.replace('--', '-'+band+'-')
    #mean_dpos = dict({'r-i':[-1.,3.], 'r-g':[3.,10.], 'r-z':[1.,7.], 'r-r':[0.,0.]})
    mean_dpos = dict({'r-i':[-1.,3.], 'r-g':[-2.,12.]})

    if frame_number_list is None:
        frame_number_list = np.linspace(0, nsamp, nframes)

    c = 0
    for num in frame_number_list:

        num = max(int(num-1), 0)

        x = 5*nbands

        # data, residual, magnitude distribution
        plt.figure(figsize=(15,10))

        for b in xrange(nbands):
            bop = int(3*b)
            psf_path = psf_basic_path.replace('.txt', bands[b]+'.txt')
            #if datatype != 'mock':
            #    psf_path = psf_path.replace('.txt', '-refit_g.txt') # only if using refit psf!
            #print 'psf_path:', psf_path
            cts_path = cts_basic_path.replace('.txt', bands[b]+'.txt')
            psf, nc, cf = get_psf_and_vals(psf_path)
            #get background
            data = np.loadtxt(cts_path)
            imsz = [len(data), len(data[0])]
            if datatype !='mock':
                data -= biases[bands[b]]
                #print biases[bands[b]]

            if b==0:
                xs = xsrcs[num]
                ys = ysrcs[num]
                if hubble_cat_path is not None:
                    hx_band, hy_band = hx, hy
            else:
                xs, ys = transform_q(xsrcs[num], ysrcs[num], pixel_transfer_mats[b-1])
                if datatype != 'mock':
                    col_name = bands[0]+'-'+bands[b]
                    #print mean_dpos[col_name]
                    xs -= mean_dpos[col_name][0]
                    ys -= mean_dpos[col_name][1]
                    if hubble_cat_path is not None:
                        #print 'pixel_transfer_mats'
                        #print pixel_transfer_mats
                        hx_band, hy_band = transform_q(hx, hy, pixel_transfer_mats[b-1])
                        hx_band -= mean_dpos[col_name][0]
                        hy_band -= mean_dpos[col_name][1]
            model = image_model_eval(xs, ys, fsrcs[b, num], bkgs[b], imsz, nc, cf)
            resid = data-model
            variance = data / gains[bands[b]]
            weight = 1. / variance
            
            np.savetxt(result_path+'/residuals/resid_'+str(num)+str(bands[b])+'.txt', resid*np.sqrt(weight))
           # plt.figure()
           # plt.imshow(resids[0,b,:,:]*np.sqrt(weight), origin='lower', interpolation='none', cmap='Greys', vmin=-30, vmax=30)
           # if boolplotsave:
           #     plt.savefig(result_path + '/residuals/resid_' + str(num) +str(bands[b])+ '.'+plttype, bbox_inches='tight')
           # plt.close()

            #plt.figure(figsize=(15,x))

            plt.subplot(nbands, 3, 1+bop)
            plt.imshow(data, origin='lower', interpolation='none', cmap='Greys', vmin=np.min(data), vmax=np.percentile(data, 95))
            plt.colorbar()
            if hubble_cat_path is not None:
                #print 'using Hubble coordinates'
                #print 'hx_band[posmask]:'
                #print hx_band[posmask]
                #print 'hmask length'
                #print len(hmask)
                #print 'hx_band[posmask][hmask]'
                #print hx_band[posmask][hmask]
                plt.scatter(hx_band[hmask], hy_band[hmask], marker='+', s=2*mag_to_cts(hf[hmask],nmgy_per_count[b])/sizefac, color='lime')
                #plt.scatter(hubble_coords[1+2*b][posmask][hmask], hubble_coords[2*b][posmask][hmask], marker='+', s=2*mag_to_cts(hf[hmask], nmgy) / sizefac, color='lime') #hubble
            else:
                mask = ref_f[:,0] > 250 # will have to change this for other data sets
                plt.scatter(ref_x[mask], ref_y[mask], marker='+', s=ref_f[:,b][mask] / sizefac, color='lime')
                mask = np.logical_not(mask)
                plt.scatter(ref_x[mask], ref_y[mask], marker='+', s=ref_f[:,b][mask] / sizefac, color='g')
            if len(data)>100:
                plt.scatter(xs, ys, marker='x', s=(len(data)**2/1e5)*fsrcs[b,num]/(2*sizefac), color='r')
            else:
                plt.scatter(xs, ys, marker='x', s=(1e4/len(data)**2)*fsrcs[b, num]/(2*sizefac), color='r')
            plt.xlim(-0.5, len(data)-0.5)
            plt.ylim(-0.5, len(data)-0.5)

            plt.subplot(nbands, 3, 2+bop)
            plt.imshow(resid*np.sqrt(weight), origin='lower', interpolation='none', cmap='Greys', vmin=-5, vmax=5)
            plt.xlim(-0.5, len(data)-0.5)
            plt.ylim(-0.5, len(data)-0.5)
            plt.colorbar()

            plt.subplot(nbands, 3, 3+bop)
            mags = adu_to_magnitude(fsrcs[b, num], nmgy_per_count[b])
            mags = mags[~np.isinf(mags)]

            (n, bins, patches) = plt.hist(ref_mags[b], histtype='step', label=labldata, color='g')
            plt.hist(mags, bins=bins, alpha=0.5, label='Chain - '+bands[b], color='r', histtype='step')
            plt.legend()
            plt.xlabel(str(bands[b]))
            plt.yscale('log')
        if boolplotsave:
            plt.savefig(result_path + '/frames/sample_' + str(num) + '_mags.'+plttype, bbox_inches='tight')
        if boolplotshow:
            plt.show()
        plt.close()

        plt.figure(figsize=(10,5))
        for b in xrange(nbands-1):
            colors = colorsrcs[b][num]
            colors = adus_to_color(fsrcs[0, num], fsrcs[b+1, num], nmgy_per_count)
            plt.subplot(1, nbands-1, b+1)
            #print 'colors:', colors
            plt.hist(colors[~np.isnan(colors)], label='Chain', bins=color_bins, bottom=0.1, histtype='step', color='r')
            plt.hist(ref_colors[b], label=labldata, bins=color_bins, bottom=0.1, histtype='step', color='g')
            plt.legend(loc=2)
            plt.yscale('log')
            plt.xlabel(bands[0] + ' - ' + bands[b+1])
        if boolplotsave and nbands >1:
            plt.savefig(result_path + '/color_histogram/color_histograms_sample_' + str(num) + '.'+plttype, bbox_inches='tight')
        if boolplotshow and nbands > 1:
            plt.show()
        plt.close()



hubble_cat_path = '/n/fink1/rfeder/mpcat/multiband_pcat/Data/idR-002583-2-0136/hubble_catalog_2583-2-0136_astrans.txt'
hubble_cat_path = '/n/fink1/rfeder/mpcat/multiband_pcat/Data/idR-002583-2-0136/hubble_pixel_coords-2583-2-0136.fits'
result_plots(result_path, ref_cat_path, datatype=datatype, burn_in_frac=0.5, bands=bands, plttype='png', hubble_cat_path=hubble_cat_path)
multiband_retro_frames(result_path, ref_cat_path, data_path, bands=bands, datatype=datatype, plttype='png', imdim=500, hubble_cat_path=hubble_cat_path)

m2plots_command = 'python m2plots.py '+dataname+' '+result_dir_name+' 1'

# UNCOMMENT TO RUN CONDENSED CATALOG CODE/COMPLETENESS/FDR
#print m2plots_command
#os.system(m2plots_command)

