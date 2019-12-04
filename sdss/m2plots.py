import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import astropy.io.fits
import astropy.wcs
import scipy.spatial
import sys
from astropy.io import fits
import time
from astropy import wcs
from scipy import interpolate
import networkx as nx
import os

print len(sys.argv)
dataname = sys.argv[1]
run_name = sys.argv[2]
make_seed_bool = int(sys.argv[3])
#if len(sys.argv)>3:
#    condensed_bool = sys.argv[3]
#else:
#    condensed_bool = 0
#sets numpy random seed
np.random.seed(25)

print 'run_name', run_name
if 'mock' in dataname:
    datatype='mock'
else:
    datatype='not mock'

chain_label = 'Two Band Catalog Ensemble'

########################## SET CONSTANTS ###################################
max_num_sources = 2000
bounds = [310.0, 410.0, 630.0, 730.0]

back = 180
gain =4.62
hwhm=2.5
nsamp=300
imdim = 100
#imdim = 500
r_nmgy = 0.00546689
sizefac = 1360.
dr = 0.5
#dmag = 0.5
dmag = 0.5
#nbins = 16
nbins = 17
minr, maxr = 15.0, 23.5
#minr, maxr = 15.5, 23.5
binw = (maxr - minr) / float(nbins)

#sets total number of iterations
iteration_number = 1
#search radius
matching_dist = 0.75
search_radius = 0.75
cut = 0.1 

########################## Setting up data paths #############################

include_hubble = 1
if datatype=='mock':
    include_hubble = 0
daophot = 1

if sys.platform=='darwin':
    base_path = '/Users/richardfeder/Documents/multiband_pcat/pcat-lion-master'
    data_path = base_path
    result_path = base_path+'/pcat-lion-results'
elif sys.platform=='linux2':
    #base_path = '/n/fink1/rfeder/mpcat/multiband_pcat'
    base_path = '/n/home07/rfederstaehle'
    result_path = '/n/home07/rfederstaehle/figures/'
    data_path = '/n/fink1/rfeder/mpcat/multiband_pcat/'
else:
    base_path = raw_input('Operating system not detected, please enter base_path directory (eg. /Users/.../pcat-lion-master):')
    if not os.path.isdir(base_path):
        raise OSError('Directory chosen does not exist. Please try again.')

##############################################################################

''' Below are three small helper functions: '''

def err_f(f):
        return 1./np.sqrt(gain*np.sum(psf0*psf0/(back+psf0*f)))
def err_mag(mag):
        f = 10**((22.5 - mag)/2.5) / r_nmgy
        return 1.08573620476 * np.sqrt((err_f(f) / f)**2 + 0.01**2)
def adutomag(adu):
        return 22.5 - 2.5 * np.log10(r_nmgy * adu)

def adus_to_color(flux0, flux1, nm_2_cts):
    colors = adu_to_magnitude(flux0, nm_2_cts[0]) - adu_to_magnitude(flux1, nm_2_cts[1])
    return colors
def adu_to_magnitude(flux, nm_2_cts):
    mags = 22.5-2.5*np.log10((np.array(flux)*nm_2_cts))
    return mags

def mag_to_cts(mags, nm_2_cts):
    flux = 10**((22.5-mags)/2.5)/nm_2_cts
    return flux


##############################################################################

def associate(a, mags_a, b, mags_b, dr, dmag, confs_b = None, sigfs_b = None):
    allmatches = a.query_ball_tree(b, dr)
    goodmatch = np.zeros(mags_a.size, np.bool)
    if confs_b is not None:
        confmatch = np.zeros(mags_a.size)
    if sigfs_b is not None:
        sigfmatch = np.zeros(mags_a.size) + float('inf')

    for i in xrange(len(allmatches)):
        matches = allmatches[i]
        if len(matches):
            mag_a = mags_a[i]
            goodmatch[i] = False
            for j in matches:
                mag_b = mags_b[j]
                if np.abs(mag_a - mag_b) < dmag:
                    goodmatch[i] = True
                    if (confs_b is not None) and (confs_b[j] > confmatch[i]):
                        confmatch[i] = confs_b[j]
                    if (sigfs_b is not None) and (sigfs_b[j] < sigfmatch[i]):
                        sigfmatch[i] = sigfs_b[j]

    if confs_b is not None:
        if sigfs_b is not None:
            return goodmatch, confmatch, sigfmatch
        else:
            return goodmatch, confmatch
    else:
        if sigfs_b is not None:
            return goodmatch, sigfmatch
        else:
            return goodmatch


###################### READ IN CATALOGS AND CATALOG CHAINS ##################################
def hubble_cat_kd():

    fitshubble = np.loadtxt(data_path+'/Data/'+dataname+'/hubble_catalog_2583-2-0136_astrans.txt')
    #fitshubble = np.loadtxt(data_path+'/Data/'+dataname+'/hubble_catalog_2583-2-0136_astrans_newr.txt')
    HTx_fits = fitshubble[:,0] - bounds[0]
    HTy_fits = fitshubble[:,1] - bounds[2]
    fitsmask = np.logical_and(fitshubble[:,2]>0, fitshubble[:,3]>0)
    fitsmask = np.logical_and(np.logical_and(np.logical_and(HTx_fits > 0+hwhm, HTx_fits < imdim-1-hwhm), np.logical_and(HTy_fits > 0+hwhm, HTy_fits < imdim-1-hwhm)), fitsmask)
    HTx_fits = HTx_fits[fitsmask]
    HTy_fits = HTy_fits[fitsmask]

    HT606 = fitshubble[:,2]
    HT814 = fitshubble[:,3]

    HT606 = HT606[fitsmask]
    HT814 = HT814[fitsmask]

    HTc = np.zeros((HTx_fits.shape[0], 2))
    HTc[:, 0] = HTx_fits
    HTc[:, 1] = HTy_fits

    HTkd = scipy.spatial.KDTree(HTc)

    print np.sum(HT606 < 22), 'HST brighter than 22'
    print np.sum(HT606 < 23), 'HST brighter than 23'
    return HTkd, HT606, HTx_fits, HTy_fits, HT814

def sdss_cat_kd(path):
    SDSS_catalog = np.loadtxt(data_path+'/Data/'+dataname+'/m2_2583.phot')
    #pulls off ra, dec, r_mag of each source
    SDSS_ra_vals = SDSS_catalog[:,4]
    SDSS_dec_vals = SDSS_catalog[:,5]
    SDSS_x_vals = SDSS_catalog[:,20]
    SDSS_y_vals = SDSS_catalog[:,21]
    SDSS_r_vals = SDSS_catalog[:,22]
    SDSS_r_error_bars = SDSS_catalog[:,23]
    SDSS_r_flux_vals = np.power(10.0, (22.5 - SDSS_r_vals)/2.5)/r_nmgy


        # cuts out our image patch of sky
    SDSS_y_vals = SDSS_y_vals[np.logical_and(SDSS_x_vals < bounds[1], SDSS_x_vals > bounds[0])]
    SDSS_r_vals = SDSS_r_vals[np.logical_and(SDSS_x_vals < bounds[1], SDSS_x_vals > bounds[0])]
    SDSS_r_error_bars = SDSS_r_error_bars[np.logical_and(SDSS_x_vals < bounds[1], SDSS_x_vals > bounds[0])]
    SDSS_r_flux_vals = SDSS_r_flux_vals[np.logical_and(SDSS_x_vals < bounds[1], SDSS_x_vals > bounds[0])]
    SDSS_ra_vals = SDSS_ra_vals[np.logical_and(SDSS_x_vals < bounds[1], SDSS_x_vals > bounds[0])]
    SDSS_dec_vals = SDSS_dec_vals[np.logical_and(SDSS_x_vals < bounds[1], SDSS_x_vals > bounds[0])]
    SDSS_x_vals = SDSS_x_vals[np.logical_and(SDSS_x_vals < bounds[1], SDSS_x_vals > bounds[0])]


    SDSS_r_vals = SDSS_r_vals[np.logical_and(SDSS_y_vals < 2092.0, SDSS_y_vals > 1992.0)]
    SDSS_r_error_bars = SDSS_r_error_bars[np.logical_and(SDSS_y_vals < 2092.0, SDSS_y_vals > 1992.0)]
    SDSS_r_flux_vals = SDSS_r_flux_vals[np.logical_and(SDSS_y_vals < 2092.0, SDSS_y_vals > 1992.0)]
    SDSS_x_vals = SDSS_x_vals[np.logical_and(SDSS_y_vals < 2092.0, SDSS_y_vals > 1992.0)]
    SDSS_ra_vals = SDSS_ra_vals[np.logical_and(SDSS_y_vals < 2092.0, SDSS_y_vals > 1992.0)]
    SDSS_dec_vals = SDSS_dec_vals[np.logical_and(SDSS_y_vals < 2092.0, SDSS_y_vals > 1992.0)]
    SDSS_y_vals = SDSS_y_vals[np.logical_and(SDSS_y_vals < 2092.0, SDSS_y_vals > 1992.0)]

    #shifts to image pixel coordinates
    SDSS_x_vals = SDSS_x_vals - bounds[0]
    SDSS_y_vals = SDSS_y_vals - 1992.0



def lion_cat_kd(path):
    lion = np.load(path)
    lion_comments = lion['comments']
    lion_n = lion['n'][-nsamp:].astype(np.int)
    lion_x = lion['x'][-nsamp:,:]
    lion_y = lion['y'][-nsamp:,:]
    lion_allf = lion['f'][:,-nsamp:]
    lion_f = lion['f'][0,-nsamp:]
    lion_nmgy = lion['nmgy']
    lion_r = adutomag(lion_f)
    print 'lion_allf_shape', lion_allf.shape
    nbands = lion['f'].shape[0]
    
    lion_fs = []
    lion_mask = (lion_f > 0) * (lion_x > 0+hwhm) * (lion_x < imdim-1-hwhm) * (lion_y > 0+hwhm) * (lion_y < imdim-1-hwhm)

    #for b in xrange(nbands):
    #    lion_fband = lion_allf[b,:,:]
    #    lion_fs.append(lion_fband)
        #mags = adu_to_magnitude(lion_allf[b,:], lion_nmgy[b])
        #lion_mags.append(mags[lion_mask])

    #lion_fs = np.array(lion_fs)
    #lion_mags = np.array(lion_mags)
    #print 'lion_fs shape:', lion_fs.shape

    #print 'lion_fs flatten shape', lion_fs.flatten().shape
    #lion_fs = lion_fs.flatten()

#    lion_mask = (lion_f > 0) * (lion_x > 0+hwhm) * (lion_x < imdim-1-hwhm) * (lion_y > 0+hwhm) * (lion_y < imdim-1-hwhm)
    lion_all = np.zeros((np.sum(lion_mask), 2))
    lion_all[:,0] = lion_x[lion_mask].flatten()
    lion_all[:,1] = lion_y[lion_mask].flatten()
    lion_r_all = lion_r[lion_mask].flatten()
    lion_kd = scipy.spatial.KDTree(lion_all)

    PCi,junk = np.mgrid[0:nsamp,0:max_num_sources]
    PCi = PCi[lion_mask].flatten()

    return lion_kd, lion_r_all, lion_x, lion_y, lion_n, lion_r, lion_all, PCi, lion_f, nbands, lion_comments, lion_allf, lion_nmgy

def mock_cat_kd(path):
    mock_catalog = np.loadtxt(path)
    mock_x = mock_catalog[:,0]
    mock_y = mock_catalog[:,1]
    mock_f = mock_catalog[:,2:]
    mock_coords = np.zeros((mock_x.shape[0], 2))
    mock_coords[:, 0] = mock_x
    mock_coords[:, 1] = mock_y
    mock_kd = scipy.spatial.KDTree(mock_coords)
    mock_rmag = adutomag(mock_f[:,0]) # r band
    #mock_rmag = adutomag(mock_f[:,1]) # i band
    print mock_rmag
    print np.sum(mock_rmag < 22), 'Mock Truth Brighter than 22'

    return mock_x, mock_kd, mock_rmag

def condensed_cat_kd(path):
    cond_cat = np.loadtxt(path)
    cond_x = cond_cat[:,0]
    cond_y = cond_cat[:,2]
    cond_r = adutomag(cond_cat[:,4])
    cond_rerr = 1.086*cond_cat[:,5]/cond_cat[:,4]
    cond_conf = cond_cat[:,8]
    # cond_s = cond_cat[:,11]
    mask = (cond_x > 0+hwhm) * (cond_x < imdim-1-hwhm) * (cond_y > 0+hwhm) * (cond_y < imdim-1-hwhm) * (cond_r > 0)

    cond_x = cond_x[mask]
    cond_y = cond_y[mask]
    cond_r = cond_r[mask]
    cond_rerr = cond_rerr[mask]
    cond_conf = cond_conf[mask]
    cond_c = np.zeros((cond_x.shape[0], 2))
    cond_c[:,0] = cond_x
    cond_c[:,1] = cond_y
    cond_kd = scipy.spatial.KDTree(cond_c)

    return cond_kd, cond_x, cond_y, cond_r


#######################################################################################

def get_completeness(test_x, test_y, test_mag, test_n, ref_x, ref_mag, ref_kd):
    complete = np.zeros((test_x.shape[0], ref_x.size))
    for i in xrange(test_x.shape[0]):
        print 'B', i
        n = test_n[i]
        CCc_one = np.zeros((n,2))
        CCc_one[:, 0] = test_x[i,0:n]
        CCc_one[:, 1] = test_y[i,0:n]
        CCmag_one = test_mag[i,0:n]

        complete[i,:] = associate(ref_kd, ref_mag, scipy.spatial.KDTree(CCc_one), CCmag_one, dr, dmag)
    complete = np.sum(complete, axis=0) / float(test_x.shape[0])
    return complete

#def generate_seed_catlaog(kd, cat_all, cat_mags_all, PCi):
def generate_seed_catalog(kd, cat_all, cat_r_all, PCi):


    matches = kd.query_ball_tree(kd, matching_dist)

    G = nx.Graph()
    G.add_nodes_from(xrange(0, cat_all.shape[0]))

    for i in xrange(cat_all.shape[0]):
        for j in matches[i]:
            if cat_r_all[i] != cat_r_all[j]:
                G.add_edge(i, j)

    current_catalogue = 0



    for i in xrange(cat_all.shape[0]):
        matches[i].sort()
        bincount = np.bincount(PCi[matches[i]]).astype(np.int)
        ending = np.cumsum(bincount).astype(np.int)
        starting = np.zeros(bincount.size).astype(np.int)
        starting[1:bincount.size] = ending[0:bincount.size-1]

        for j in xrange(bincount.size):
            if j == PCi[i]: # do not match to same catalogue
                continue
            if bincount[j] == 0: # no match to catalog j
                continue
            if bincount[j] == 1: # exactly one match to catalog j
                continue
            if bincount[j] > 1:
                dist2 = matching_dist**2
                l = -1
                for k in xrange(starting[j], ending[j]):
                    m = matches[i][k]
                    newdist2 = np.sum((cat_all[i,:] - cat_all[m,:])**2)
                    if newdist2 < dist2:
                        l = m
                        dist2 = newdist2

                if l == -1:
                    print " didn't find edge even though multiple matches from this catalogue?"
                for k in xrange(starting[j], ending[j]):
                    m = matches[i][k]
                    if m != l:
                        if G.has_edge(i, m):
                            G.remove_edge(i, m)
                            # print "killed", i, m

    seeds = []

    while nx.number_of_nodes(G) > 0:
        deg = nx.degree(G)
        i = max(deg, key=deg.get)
        neighbors = nx.all_neighbors(G, i)
        print 'found', i
        seeds.append([cat_all[i, 0], cat_all[i, 1], deg[i]])
        G.remove_node(i)
        G.remove_nodes_from(neighbors)

    seeds = np.array(seeds)
    print seeds
    np.savetxt(base_path+'/pcat-lion-results/'+run_name+'/seeds.txt', seeds)
    np.savetxt(result_path+'/'+run_name+'/seeds.txt', seeds)
    return seeds



def clusterize(seed_cat, cat_x, cat_y, cat_n, cat_r, cat_fs, nmgy):

    nbands = cat_fs.shape[0]
    sorted_posterior_sample = np.zeros((nsamp, max_num_sources, 2+nbands))
    print cat_x.shape, cat_r.shape
    print cat_fs.shape

    #nbands = cat_fs.shape[0]
    print 'nbands=', nbands
    print 
    for i in xrange(nsamp):
        #cat = np.zeros((max_num_sources, 3))
        cat = np.zeros((max_num_sources, 2+nbands))
        cat[:,0] = cat_x[i,:]
        cat[:,1] = cat_y[i,:]
        for b in xrange(nbands):
            cat[:,2+b] = cat_fs[b,i,:]
        #cat[:,2] = cat_r[i,:]
        cat = np.flipud(cat[cat[:,2].argsort()])
        sorted_posterior_sample[i] = cat

    PCx = sorted_posterior_sample[:,:,0]
    PCy = sorted_posterior_sample[:,:,1]
    PCf = sorted_posterior_sample[:,:,2:]

    stack = np.zeros((np.sum(cat_n), 2))
    j = 0
    for i in xrange(cat_n.size): # don't have to for loop to stack but oh well
        n = cat_n[i]
        stack[j:j+n, 0] = PCx[i, 0:n]
        stack[j:j+n, 1] = PCy[i, 0:n]
        j += n

    #creates tree, where tree is Pcc_stack
    tree = scipy.spatial.KDTree(stack)

    #keeps track of the clusters
    #clusters = np.zeros((nsamp, len(seed_cat)*3))
    clusters = np.zeros((nsamp, len(seed_cat)*(2+nbands)))

    #numpy mask for sources that have been matched
    mask = np.zeros(len(stack))

    #first, we iterate over all sources in the seed catalog:
    for ct in range(0, len(seed_cat)):

        #query ball point at seed catalog
        matches = tree.query_ball_point(seed_cat[ct], search_radius)

        #in each catalog, find first instance of match w/ desired source (-- first instance okay b/c every catalog is sorted brightest to faintest)
        ##this for loop should naturally populate original seed catalog as well! (b/c all distances 0)
        for i in range(0, nsamp):
            #for specific catalog, find indices of start and end within large tree
            cat_lo_ndx = np.sum(cat_n[:i])
            cat_hi_ndx = np.sum(cat_n[:i+1])
            #want in the form of a numpy array so we can use array slicing/masking  
            matches = np.array(matches)

            #find the locations of matches to ct within specific catalog i
            culled_matches =  matches[np.logical_and(matches >= cat_lo_ndx, matches < cat_hi_ndx)] 

            if culled_matches.size > 0:
                #cut according to mask
                culled_matches = culled_matches[mask[culled_matches] == 0]
                    #if there are matches remaining, we then find the brightest and update
                if culled_matches.size > 0:
                        #find brightest
                    match = np.min(culled_matches)

                    #flag it in the mask
                    mask[match] += 1

                    #find x, y, flux of match
                    x = PCx[i,match-cat_lo_ndx]
                    y = PCy[i,match-cat_lo_ndx]
                    #f = PCf[i,match-cat_lo_ndx]
                    #f = PCf[:,i,match-cat_lo_ndx]
                    #add information to cluster array
                    clusters[i][ct] = x
                    clusters[i][len(seed_cat)+ct] = y
                    #clusters[i][2*len(seed_cat)+ct] = f
                    for b in xrange(nbands):
                        clusters[i][(2+b)*len(seed_cat)+ct] = PCf[i,match-cat_lo_ndx, b]

    #we now generate a CLASSICAL CATALOG from clusters
    cat_len = len(seed_cat)

    #arrays to store 'classical' catalog parameters
    mean_x = np.zeros(cat_len)
    mean_y = np.zeros(cat_len)
    mean_f = np.zeros(cat_len)
    #mean_mag = np.zeros(cat_len)
    mean_mags = np.zeros(shape=(nbands, cat_len))
    err_x = np.zeros(cat_len)
    err_y = np.zeros(cat_len)
    err_f = np.zeros(cat_len)
    #err_mag = np.zeros(cat_len)
    err_mags = np.zeros(shape=(nbands,cat_len))
    confidence = np.zeros(cat_len)

    #confidence interval defined for err_(x,y,f)
    hi = 84
    lo = 16

    for i in range(0, len(seed_cat)):
        x = clusters[:,i][np.nonzero(clusters[:,i])]
        y = clusters[:,i+cat_len][np.nonzero(clusters[:,i+cat_len])]
        #f = clusters[:,i+2*cat_len][np.nonzero(clusters[:,i+2*cat_len])]
        
        fs=[]
        for b in xrange(nbands):
            fs.append(clusters[:,i+(2+b)*cat_len][np.nonzero(clusters[:,i+(2+b)*cat_len])])
        
        assert x.size == y.size
        #assert x.size == f.size
        assert x.size == fs[0].size
        confidence[i] = x.size/float(nsamp)

        mean_x[i] = np.mean(x)
        mean_y[i] = np.mean(y)
        #mean_f[i] = np.mean(f)
        mean_f[i] = np.mean(fs[0])
        #mean_mag[i] = 22.5 - 2.5*np.log10(np.mean(f)*r_nmgy)

        for b in xrange(nbands):
            mean_mags[b,i] = adu_to_magnitude(np.mean(fs[b]), nmgy[b])

        if x.size > 1:
            err_x[i] = np.percentile(x, hi) - np.percentile(x, lo)
            err_y[i] = np.percentile(y, hi) - np.percentile(y, lo)
            #err_f[i] = np.percentile(f, hi) - np.percentile(f, lo)
            err_f[i] = np.percentile(fs[0], hi) - np.percentile(fs[0],lo)
            #err_mag[i] = np.absolute( ( 22.5 - 2.5*np.log10(np.percentile(f, hi)*r_nmgy) )  - ( 22.5 - 2.5*np.log10(np.percentile(f, lo)*r_nmgy) ) )
            for b in xrange(nbands):
                err_mags[b,i] = np.absolute(adu_to_magnitude(np.percentile(fs[b], hi), nmgy[b])-adu_to_magnitude(np.percentile(fs[b], lo), nmgy[b]))
    #makes classical catalog
#    classical_catalog = np.zeros( (cat_len, 9) ) 

    classical_catalog = np.zeros((cat_len, 7+2*nbands))
    classical_catalog[:,0] = mean_x
    classical_catalog[:,1] = err_x
    classical_catalog[:,2] = mean_y
    classical_catalog[:,3] = err_y
    classical_catalog[:,4] = mean_f
    classical_catalog[:,5] = err_f
    classical_catalog[:,6] = confidence

    for b in xrange(nbands):
        classical_catalog[:,7+2*b] = mean_mags[b]
        classical_catalog[:,8+2*b] = err_mags[b]

#    classical_catalog[:,6] = mean_mag
#    classical_catalog[:,7] = err_mag
#    classical_catalog[:,8] = confidence

    #saves catalog
    np.savetxt(base_path+'/pcat-lion-results/'+run_name+'/classical_catalog.txt', classical_catalog)
    np.savetxt(base_path+'/figures/'+run_name+'/classical_catalog.txt', classical_catalog)
    pix_offset = 0.5

    return classical_catalog




#------------------- PORTILLO ET AL 2017 -----------------------

if datatype != 'mock':
    try:
        PCcat = np.loadtxt(data_path+'Data/'+dataname+'/posterior_sample.txt')
        maxn = 3000
        PCn = PCcat[-nsamp:,10003].astype(np.int)
        PCx = PCcat[-nsamp:,10004:10004+maxn]
        PCy = PCcat[-nsamp:,10004+maxn:10004+2*maxn]
        PCf = PCcat[-nsamp:,10004+2*maxn:10004+3*maxn]
        print PCn.shape, PCx.shape, PCy.shape, PCf.shape
        PCr = adutomag(PCf)
        mask = (PCf > 0) * (PCx > 0+hwhm) * (PCx < 99-hwhm) * (PCy > 0+hwhm) * (PCy < 99-hwhm)
        PCc_all = np.zeros((np.sum(mask), 2))
        PCc_all[:, 0] = PCx[mask].flatten()
        PCc_all[:, 1] = PCy[mask].flatten()
        PCr_all = PCr[mask].flatten()
        PCkd = scipy.spatial.KDTree(PCc_all)
        print np.mean(PCn), 'mean PCAT sources from Portillo et al. 2017'
    except:
        print 'Nope'



#----------------- READ IN CATALOGS AND CREATE KD TREES ------------------

if datatype == 'mock':
    mock_x, mock_kd, mock_rmag = mock_cat_kd(data_path+'/Data/'+dataname+'/truth/'+dataname+'-tru.txt')

if include_hubble:
    HTkd, HT606, HTx_fits, HTy_fits, HT814 = hubble_cat_kd()


# load in chain
lion_kd, lion_r_all, lion_x, lion_y, lion_n, lion_r, lion_all, PCi, lion_f, nb, lion_comments, lion_fs, nmgy = lion_cat_kd(base_path+'/pcat-lion-results/'+run_name+'/chain.npz')

print 'lion_r_shape here is', lion_r.shape
print 'lion_f shape is', lion_f.shape
print 'lion_r+all shape is', lion_r_all.shape

if datatype != 'mock':
    if not make_seed_bool and os.path.isfile(base_path+'/pcat-lion-results/'+run_name+'/seeds.txt'):
        dat = np.loadtxt(base_path+'/pcat-lion-results/'+run_name+'/seeds.txt')
    else:
        dat = generate_seed_catalog(lion_kd, lion_all, lion_r_all, PCi)    
    cut = 0.1

    # plots histogram of confidence
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1,2,2)
    plt.hist(dat[:,2], bins=50)
    plt.xlabel("Number of Samples")
    plt.ylim([0, 1250])
    plt.title("Seed Catalog")
    plt.subplot(1,2,1)
    plt.hist(dat[:,2]/nsamp, bins=50)
    plt.xlabel("Prevalence")
    plt.ylim([0, 1250])
    plt.title("Seed Catalog")
    plt.savefig(base_path+'/pcat-lion-results/'+run_name+'/hist_seed_cat.pdf')
    plt.savefig(result_path+'/'+run_name+'/hist_seed_cat.pdf')
    #performs confidence cut
    x = dat[:,0][dat[:,2] > cut*nsamp]
    y = dat[:,1][dat[:,2] > cut*nsamp]
    n = dat[:,2][dat[:,2] > cut*nsamp]

    assert x.size == y.size
    assert x.size == n.size

    seed_cat = np.zeros((x.size, 2))
    seed_cat[:,0] = x
    seed_cat[:,1] = y
    cat_len = x.size

    condensed_cat = clusterize(seed_cat, lion_x, lion_y, lion_n, lion_f, lion_fs, nmgy)
    condensed_x = condensed_cat[:,0]
    condensed_y = condensed_cat[:,1]

    print 'min, max of x and y:'
    print np.amin(condensed_x), np.amax(condensed_x)
    print np.amin(condensed_y), np.amax(condensed_y)

    cond_kd, cond_x, cond_y, cond_r = condensed_cat_kd(base_path+'/pcat-lion-results/'+run_name+'/classical_catalog.txt')

    print np.amin(cond_x), np.amax(cond_x)
    print np.amax(cond_y), np.amax(cond_y)

    # # plots histogram of confidence

    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1,2,2)
    plt.hist(condensed_cat[:,6], bins=50)
    plt.xlabel("Number of Samples")
    plt.ylim([0, 1250])
    plt.title("Condensed Catalog")
    plt.subplot(1,2,1)
    plt.hist(condensed_cat[:,6]/nsamp, bins=50)
    plt.xlabel("Prevalence")
    plt.ylim([0, 1250])
    plt.title("Condensed Catalog")
    plt.savefig(base_path+'/pcat-lion-results/'+run_name+'/hist_classical_cat.pdf')


prec_portillo17, prec_lion, prec_condensed = [np.zeros(nbins) for x in xrange(3)]

if datatype == 'mock':
    goodmatch_lion = associate(lion_kd, lion_r_all, mock_kd, mock_rmag, dr, dmag)
else:

    #goodmatch_portillo17 = associate(PCkd, PCr_all, HTkd, HT814, dr, dmag)
    #goodmatch_lion = associate(lion_kd, lion_r_all, HTkd, HT814, dr, dmag)
    #goodmatch_condensed = associate(cond_kd, cond_r, HTkd, HT814, dr, dmag)
    goodmatch_portillo17 = associate(PCkd, PCr_all, HTkd, HT606, dr, dmag)
    goodmatch_lion = associate(lion_kd, lion_r_all, HTkd, HT606, dr, dmag)
    goodmatch_condensed = associate(cond_kd, cond_r, HTkd, HT606, dr, dmag)

for i in xrange(nbins):
    rlo = minr + i * binw
    rhi = rlo + binw

    inbin = np.logical_and(lion_r_all >= rlo, lion_r_all < rhi)
    prec_lion[i] = np.sum(np.logical_and(inbin, goodmatch_lion)) / float(np.sum(inbin))
    #print rlo, rhi, np.sum(np.logical_and(inbin, goodmatch_lion))
    if datatype != 'mock':
        inbin = np.logical_and(PCr_all >= rlo, PCr_all < rhi)
        inbin_cond = np.logical_and(cond_r >= rlo, cond_r < rhi)
        prec_portillo17[i] = np.sum(np.logical_and(inbin, goodmatch_portillo17)) / float(np.sum(inbin))
        prec_condensed[i] = np.sum(np.logical_and(inbin_cond, goodmatch_condensed)) / float(np.sum(inbin_cond))
        print rlo, rhi, np.sum(np.logical_and(inbin_cond, goodmatch_condensed))
print 'prec_lion', prec_lion
print 'prec_condensed', prec_condensed

chain_label = str(nb)+' Band Catalog Ensemble'
plt.figure()
if datatype != 'mock':
    label = 'Portillo et al. (2017)'
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-prec_portillo17, c='r', label=label, marker='x', markersize=10, mew=2)
else:
    label = 'Mock'
plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-prec_lion, c='b', label=chain_label, marker='+', markersize=10, mew=2)
plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-prec_condensed, c='y', label='Condensed Catalog', marker='+', markersize=10, mew=2)
plt.xlabel('SDSS r magnitude')
plt.ylabel('false discovery rate')
plt.ylim((-0.05, 0.9))
plt.xlim((15,24))
plt.legend(prop={'size':12}, loc = 'best')
#plt.savefig(result_path+'/'+run_name+'/fdr-lion'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'.pdf')
plt.savefig(result_path+'/'+run_name+'/fdr-lion'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'-15.0-23.5.pdf')

plt.close()

fdr_lion = 1-prec_lion
fdr_lion_condensed = 1-prec_condensed
fdr_portillo17 = 1-prec_portillo17
np.savetxt(result_path+'/'+run_name+'/fdr_'+str(dr)+'_'+str(dmag)+'_condensed.txt', fdr_lion_condensed)
np.savetxt(result_path+'/'+run_name+'/fdr_'+str(dr)+'_'+str(dmag)+'.txt', fdr_lion)
np.savetxt(result_path+'/'+run_name+'/fdr_portillo17.txt', fdr_portillo17)


if datatype != 'mock':
    #complete_lion = get_completeness(lion_x, lion_y, lion_r, lion_n, HTx_fits, HT814, HTkd)
    #complete_portillo17 = get_completeness(PCx, PCy, PCr, PCn, HTx_fits, HT814, HTkd)
    #complete_condensed = associate(HTkd, HT814, cond_kd, cond_r, dr, dmag)

    complete_lion = get_completeness(lion_x, lion_y, lion_r, lion_n, HTx_fits, HT606, HTkd)
    complete_portillo17 = get_completeness(PCx, PCy, PCr, PCn, HTx_fits, HT606, HTkd)
    complete_condensed = associate(HTkd, HT606, cond_kd, cond_r, dr, dmag)
    # complete_daophot = associate(HTkd, HT606, daophot_kd, daophot_r, dr, dmag)
else:
    print mock_rmag.shape
    complete_lion = get_completeness(lion_x, lion_y, lion_r, lion_n, mock_x, mock_rmag, mock_kd)


reclPC_portillo17 = np.zeros(nbins)
reclPC_lion= np.zeros(nbins)
recl_cond = np.zeros(nbins)
recl_daophot = np.zeros(nbins)



for i in xrange(nbins):
    rlo = minr + i * binw
    rhi = rlo + binw
    if datatype != 'mock':
        #inbin = np.logical_and(HT814 >= rlo, HT814 < rhi)
        inbin = np.logical_and(HT606 >= rlo, HT606 < rhi)
        reclPC_portillo17[i] = np.sum(complete_portillo17[inbin]) / float(np.sum(inbin))
        recl_cond[i] = np.sum(complete_condensed[inbin])/float(np.sum(inbin))
    else:
        inbin = np.logical_and(mock_rmag >= rlo, mock_rmag< rhi)
    reclPC_lion[i] = np.sum(complete_lion[inbin]) / float(np.sum(inbin))
    #recl_cond[i] = np.sum(complete_condensed[inbin])/float(np.sum(inbin))
    print rlo, rhi, np.sum(complete_lion[inbin])
    #print np.sum(complete_condensed[inbin])


plt.figure()
plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC_lion, c='b', label=chain_label, marker='+', markersize=10, mew=2)
if datatype != 'mock':
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC_portillo17, c='r', label=label, marker='x', markersize=10, mew=2)
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, recl_cond, c='y', label='Condensed Catalog', marker='x', markersize=10, mew=2)
plt.xlabel('HST F606W magnitude', fontsize='large')
plt.ylabel('completeness', fontsize='large')
plt.ylim((-0.1,1.1))
plt.legend(loc='best', fontsize='large')
#plt.savefig(result_path+'/'+run_name+'/completeness-lion'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'.pdf')
plt.savefig(result_path+'/'+run_name+'/completeness-lion'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'-15.0-23.5.pdf')

plt.close()
np.savetxt(result_path+'/'+run_name+'/completeness_'+str(dr)+'_'+str(dmag)+'_condensed.txt', recl_cond)
np.savetxt(result_path+'/'+run_name+'/completeness_'+str(dr)+'_'+str(dmag)+'.txt', reclPC_lion)
np.savetxt(result_path+'/'+run_name+'/completeness_portillo.txt', reclPC_portillo17)
print 'lion comments:', lion_comments
with open(result_path+'/'+run_name+'/comments.txt', 'w') as p:
    p.write(lion_comments)
    p.close()


