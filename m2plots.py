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


dataname = sys.argv[1]
run_name = sys.argv[2]

#sets numpy random seed
np.random.seed(25)

print 'run_name', run_name
if 'mock' in dataname:
    datatype='mock'
else:
    datatype='not mock'

chain_label = 'Two Band Catalog Ensemble'

########################## SET CONSTANTS ###################################
max_num_sources = 2500
bounds = [310.0, 410.0, 630.0, 730.0]

back = 180
gain =4.62
hwhm=2.5
nsamp=50
imdim = 100
r_nmgy = 0.00546689
sizefac = 1360.
dr = 0.5
dmag = 0.5
nbins = 16
minr, maxr = 15.5, 23.5
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
elif sys.platform=='linux2':
    #base_path = '/n/fink1/rfeder/mpcat/multiband_pcat'
    base_path = '/n/home07/rfederstaehle/'
    result_path = '/n/home07/rfederstaehle/figures/'
    data_path = '/n/fink1/rfeder/mpcat/multiband_pcat'
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
    HTx_fits = fitshubble[:,0] - bounds[0]
    HTy_fits = fitshubble[:,1] - bounds[2]
    fitsmask = np.logical_and(fitshubble[:,2]>0, fitshubble[:,3]>0)
    fitsmask = np.logical_and(np.logical_and(np.logical_and(HTx_fits > 0+hwhm, HTx_fits < 99-hwhm), np.logical_and(HTy_fits > 0+hwhm, HTy_fits < 99-hwhm)), fitsmask)
    HTx_fits = HTx_fits[fitsmask]
    HTy_fits = HTy_fits[fitsmask]

    HT606 = fitshubble[:,2]
    HT814 = fitshubble[:,3]

    HTcat = np.loadtxt(data_path+'/Data/NGC7089R.RDVIQ.cal.adj.zpt', skiprows=1)
    HT606 = HT606[fitsmask]
    HT814 = HT814[fitsmask]

    HTc = np.zeros((HTx_fits.shape[0], 2))
    HTc[:, 0] = HTx_fits
    HTc[:, 1] = HTy_fits
    HTkd = scipy.spatial.KDTree(HTc)

    print np.sum(HT606 < 22), 'HST brighter than 22'

    return HTkd, HT606, HTx_fits, HTy_fits

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
    lion_n = lion['n'][-nsamp:].astype(np.int)
    lion_x = lion['x'][-nsamp:,:]
    lion_y = lion['y'][-nsamp:,:]
    lion_f = lion['f'][0,-nsamp:]
    lion_r = adutomag(lion_f)
    print 'lion r-band magnitudes:', lion_r

    lion_mask = (lion_f > 0) * (lion_x > 0+hwhm) * (lion_x < imdim-1-hwhm) * (lion_y > 0+hwhm) * (lion_y < imdim-1-hwhm)
    lion_all = np.zeros((np.sum(lion_mask), 2))
    lion_all[:,0] = lion_x[lion_mask].flatten()
    lion_all[:,1] = lion_y[lion_mask].flatten()
    lion_r_all = lion_r[lion_mask].flatten()
    lion_kd = scipy.spatial.KDTree(lion_all)

    return lion_kd, lion_r_all, lion_x, lion_y, lion_n, lion_r

def mock_cat_kd(path):
    mock_catalog = np.loadtxt(path)
    mock_x = mock_catalog[:,0]
    mock_y = mock_catalog[:,1]
    mock_f = mock_catalog[:,2:]
    mock_coords = np.zeros((mock_x.shape[0], 2))
    mock_coords[:, 0] = mock_x
    mock_coords[:, 1] = mock_y
    mock_kd = scipy.spatial.KDTree(mock_coords)
    mock_rmag = adutomag(mock_f[:,0])
    print mock_rmag
    print np.sum(mock_rmag < 22), 'Mock Truth Brighter than 22'

    return mock_x, mock_kd, mock_rmag



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


def generate_seed_catalog(path, kd, cat_all, cat_r_all):

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
        bincount = np.bincount(cat_r_all[matches[i]]).astype(np.int)
        ending = np.cumsum(bincount).astype(np.int)
        starting = np.zeros(bincount.size).astype(np.int)
        starting[1:bincount.size] = ending[0:bincount.size-1]

        for j in xrange(bincount.size):
            if j == cat_r_all[i]: # do not match to same catalogue
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
                            print "killed", i, m

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
    #np.savetxt(data_path+'/Data/'+)
    return seeds



def clusterize(seed_cat, cat_n, stack, colors):

    #creates tree, where tree is Pcc_stack
    tree = scipy.spatial.KDTree(stack)

    #keeps track of the clusters
    clusters = np.zeros((nsamp, len(seed_cat)*3))

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
                        x = PCx[i][match-cat_lo_ndx]
                        y = PCy[i][match-cat_lo_ndx]
                        f = PCf[i][match-cat_lo_ndx]

                        #add information to cluster array
                        clusters[i][ct] = x
                        clusters[i][len(seed_cat)+ct] = y
                        clusters[i][2*len(seed_cat)+ct] = f

    #we now generate a CLASSICAL CATALOG from clusters
    cat_len = len(seed_cat)

    #arrays to store 'classical' catalog parameters
    mean_x = np.zeros(cat_len)
    mean_y = np.zeros(cat_len)
    mean_f = np.zeros(cat_len)
    mean_mag = np.zeros(cat_len)
    err_x = np.zeros(cat_len)
    err_y = np.zeros(cat_len)
    err_f = np.zeros(cat_len)
    err_mag = np.zeros(cat_len)
    confidence = np.zeros(cat_len)

    #confidence interval defined for err_(x,y,f)
    hi = 84
    lo = 16

    for i in range(0, len(seed_cat)):
        x = clusters[:,i][np.nonzero(clusters[:,i])]
        y = clusters[:,i+cat_len][np.nonzero(clusters[:,i+cat_len])]
        f = clusters[:,i+2*cat_len][np.nonzero(clusters[:,i+2*cat_len])]

        assert x.size == y.size
        assert x.size == f.size
            confidence[i] = x.size/300.0

        mean_x[i] = np.mean(x)
        mean_y[i] = np.mean(y)
        mean_f[i] = np.mean(f)
        mean_mag[i] = 22.5 - 2.5*np.log10(np.mean(f)*r_nmgy)

        if x.size > 1:

            err_x[i] = np.percentile(x, hi) - np.percentile(x, lo)
            err_y[i] = np.percentile(y, hi) - np.percentile(y, lo)
            err_f[i] = np.percentile(f, hi) - np.percentile(f, lo)
            err_mag[i] = np.absolute( ( 22.5 - 2.5*np.log10(np.percentile(f, hi)*r_nmgy) )  - ( 22.5 - 2.5*np.log10(np.percentile(f, lo)*r_nmgy) ) )

    #makes classical catalog
    classical_catalog = np.zeros( (cat_len, 9) )
    classical_catalog[:,0] = mean_x
    classical_catalog[:,1] = err_x
    classical_catalog[:,2] = mean_y
    classical_catalog[:,3] = err_y
    classical_catalog[:,4] = mean_f
    classical_catalog[:,5] = err_f
    classical_catalog[:,6] = mean_mag
    classical_catalog[:,7] = err_mag
    classical_catalog[:,8] = confidence

    #saves catalog
    np.savetxt('/n/home12/blee/code/stephen/pcat-dnest/Data/classical_catalog_iter_', classical_catalog)
    pix_offset = 0.5

    return classical_catalog




#------------------- PORTILLO ET AL 2017 -----------------------

if datatype != 'mock':
    PCcat = np.loadtxt(data_path+'/Data/posterior_sample.txt')
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



#----------------- READ IN CATALOGS AND CREATE KD TREES ------------------

if datatype == 'mock':
    mock_x, mock_kd, mock_rmag = mock_cat_kd(data_path+'/Data/'+dataname+'/truth/'+dataname+'-tru.txt')

if include_hubble:
    HTkd, HT606, HTx_fits, HTy_fits = hubble_cat_kd()


# load in chain
lion_kd, lion_r_all, lion_x, lion_y, lion_n, lion_r, lion_all = lion_cat_kd(base_path+'/pcat-lion-results/'+run_name+'/chain.npz')

#seed_cat = generate_seed_catalog(path, lion_kd, lion_all, lion_r_all)

prec_portillo17, prec_lion = [np.zeros(nbins) for x in xrange(2)]



if datatype == 'mock':
    goodmatch_lion = associate(lion_kd, lion_r_all, mock_kd, mock_rmag, dr, dmag)
else:
    goodmatch_portillo17 = associate(PCkd, PCr_all, HTkd, HT606, dr, dmag)
    goodmatch_lion = associate(lion_kd, lion_r_all, HTkd, HT606, dr, dmag)


for i in xrange(nbins):
    rlo = minr + i * binw
    rhi = rlo + binw

    inbin = np.logical_and(lion_r_all >= rlo, lion_r_all < rhi)
    prec_lion[i] = np.sum(np.logical_and(inbin, goodmatch_lion)) / float(np.sum(inbin))

    if datatype != 'mock':
        inbin = np.logical_and(PCr_all >= rlo, PCr_all < rhi)
        prec_portillo17[i] = np.sum(np.logical_and(inbin, goodmatch_portillo17)) / float(np.sum(inbin))

print 'prec_lion', prec_lion


plt.figure()
if datatype != 'mock':
    label = 'Portillo et al. (2017)'
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-prec_portillo17, c='r', label=label, marker='x', markersize=10, mew=2)
else:
    label = 'Mock'
plt.plot(minr + (np.arange(nbins)+0.5)*binw, 1-prec_lion, c='b', label=chain_label, marker='+', markersize=10, mew=2)
plt.xlabel('SDSS r magnitude')
plt.ylabel('false discovery rate')
plt.ylim((-0.05, 0.9))
plt.xlim((15,24))
plt.legend(prop={'size':12}, loc = 'best')
plt.savefig(result_path+'/'+run_name+'/fdr-lion'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'.pdf')
plt.close()

fdr_lion = 1-prec_lion
np.savetxt(result_path+'/'+run_name+'/fdr_'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'.txt', fdr_lion)



if datatype != 'mock':
    complete_lion = get_completeness(lion_x, lion_y, lion_r, lion_n, HTx_fits, HT606, HTkd)
    complete_portillo17 = get_completeness(PCx, PCy, PCr, PCn, HTx_fits, HT606, HTkd)
else:
    print mock_rmag.shape
    complete_lion = get_completeness(lion_x, lion_y, lion_r, lion_n, mock_x, mock_rmag, mock_kd)


reclPC_portillo17 = np.zeros(nbins)
reclPC_lion= np.zeros(nbins)



for i in xrange(nbins):
    rlo = minr + i * binw
    rhi = rlo + binw
    if datatype != 'mock':
        inbin = np.logical_and(HT606 >= rlo, HT606 < rhi)
        reclPC_portillo17[i] = np.sum(complete_portillo17[inbin]) / float(np.sum(inbin))
    else:
        inbin = np.logical_and(mock_rmag >= rlo, mock_rmag< rhi)
    reclPC_lion[i] = np.sum(complete_lion[inbin]) / float(np.sum(inbin))



plt.figure()
plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC_lion, c='b', label=chain_label, marker='+', markersize=10, mew=2)
if datatype != 'mock':
    plt.plot(minr + (np.arange(nbins)+0.5)*binw, reclPC_portillo17, c='r', label=label, marker='x', markersize=10, mew=2)
plt.xlabel('HST F606W magnitude', fontsize='large')
plt.ylabel('completeness', fontsize='large')
plt.ylim((-0.1,1.1))
plt.legend(loc='best', fontsize='large')
plt.savefig(result_path+'/'+run_name+'/completeness-lion'+str(run_name)+'_'+str(dr)+'_'+str(dmag)+'.pdf')
plt.close()

np.savetxt(result_path+'/'+run_name+'/completeness_'+str(run_name)+'.txt', reclPC_lion)





# sorted_posterior_sample = np.zeros( (len(posterior_sample), max_num_sources, 3) )

# #first, we must sort each sample in decreasing order of brightness
# for i in range(0, len(posterior_sample)):

#     cat = np.zeros( (max_num_sources, 3) )
#     cat[:,0] = posterior_x_vals[i]
#     cat[:,1] = posterior_y_vals[i]
#     cat[:,2] = posterior_r_flux_vals[i] 

#     cat = np.flipud( cat[cat[:,2].argsort()] )

#     sorted_posterior_sample[i] = cat

# print "time, sorting: " + str(time.clock() - start_time)


# lion_stack = np.zeros((np.sum(lion_n), 2))
# j = 0
# for i in xrange(PCn.size): # don't have to for loop to stack but oh well
#     n = lion_n[i]
#     lion_stack[j:j+n, 0] = lion_x[i, 0:n]
#     lion_stack[j:j+n, 1] = lion_y[i, 0:n]
#     j += n


# # plots histogram of confidence

# fig = plt.gcf()
# fig.set_size_inches(10, 5)
# plt.subplots_adjust(wspace=0.5)


# plt.subplot(1,2,2)
# plt.hist(dat[:,2], bins=50)
# plt.xlabel("Number of Samples")
# plt.ylim([0, 1250])
# plt.title("Seed Catalog")

# plt.subplot(1,2,1)
# plt.hist(dat[:,2]/nsamp, bins=50)
# plt.xlabel("Prevalence")
# plt.ylim([0, 1250])
# plt.title("Seed Catalog")

# plt.savefig('/n/home12/blee/output/m2/iter/hist_seed_cat.pdf')

# print np.sum(dat[:,2] == 2.0) 


# #performs confidence cut
# x = dat[:,0][dat[:,2] > cut*nsamp]
# y = dat[:,1][dat[:,2] > cut*nsamp]
# n = dat[:,2][dat[:,2] > cut*nsamp]

# assert x.size == y.size
# assert x.size == n.size

# seed_cat = np.zeros((x.size, 2))
# seed_cat[:,0] = x
# seed_cat[:,1] = y
# cat_len = x.size

# print "HIT1"

# colors = []
# for i in range(0, len(seed_cat)):
#     colors.append(np.random.rand(3,1))

# classical_catalog = clusterize(seed_cat, colors, 0)


# print "HIT"


# # plots histogram of confidence

# fig = plt.gcf()
# fig.set_size_inches(10, 5)
# plt.subplots_adjust(wspace=0.5)


# plt.subplot(1,2,2)
# plt.hist(classical_catalog[:,8], bins=50)
# plt.xlabel("Number of Samples")
# plt.ylim([0, 1250])
# plt.title("Condensed Catalog")

# plt.subplot(1,2,1)
# plt.hist(classical_catalog[:,8]/nsamp, bins=50)
# plt.xlabel("Prevalence")
# plt.ylim([0, 1250])
# plt.title("Condensed Catalog")


# plt.savefig('/n/home12/blee/output/m2/iter/hist_classical_cat.pdf')


# sys.exit()



# #
# plt.clf()
# plt.scatter(classical_catalog[:,6], classical_catalog[:,8], marker='x')
# plt.title("Search Radius: " + str(search_radius) + ", Iteration: 1")
# plt.savefig('/n/home12/blee/output/m2/iter/classical_scatter_0')
# #

# #stores confidence sum for each iteration
# confidence_sum_ra = np.zeros(iteration_number)

# print "time elapsed, 0: " + str(time.clock() - start_time)
# print "Confidence sum: " + str(np.sum(classical_catalog[:,8])) #str(np.sum(classical_catalog[:,6][classical_catalog[:8] > 0.5))
# confidence_sum_ra[0] = str(np.sum(classical_catalog[:,8]))

# for i in range(0, iteration_number - 1):

#     thinned_cat = np.zeros((cat_len, 2))
#     thinned_cat[:,0] = classical_catalog[:,0]
#     thinned_cat[:,1] = classical_catalog[:,2]

#     classical_catalog = clusterize(thinned_cat, colors, i+1)

#     #
#     plt.clf()
#     plt.scatter(classical_catalog[:,6], classical_catalog[:,8], marker='x')
#     plt.title("Search Radius: " + str(search_radius) + ", Iteration: " + str(i+2))
#     plt.savefig('/n/home12/blee/output/m2/iter/classical_scatter_' + str(i+1))
#     #

#         print "time elapsed, " + str(i+1) + ": " + str(time.clock() - start_time)
#     print "Confidence sum: " + str(np.sum(classical_catalog[:,8])) #str(np.sum(classical_catalog[:,6][classical_catalog[:8] > 0.5))
#     confidence_sum_ra[i+1] = str(np.sum(classical_catalog[:,8]))

# np.savetxt('/n/home12/blee/code/stephen/pcat-dnest/Data/classical_catalog_iter', classical_catalog)



    
