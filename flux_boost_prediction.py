''' 
Lets look at stephens calculation for the delta log likelihood between sources and see what the range of
source configurations is that gets properly deblended

'''

from scipy.optimize import fsolve
import numpy as np

def overlap_fn(x, d, f1, f2, s):
    return f2*np.exp(-0.25*(d-x)*(d-x)/(s*s))*(d-x) -f1*np.exp(-0.25*x*x/(s*s))*x

def bestfitpos(d, f1, f2, s):
    assert f1 >= f2
    return fsolve(overlap_fn, 0, args=(d, f1, f2, s))

def bestfitflux(d, f1, f2, s):
    x = bestfitpos(d, f1, f2, s)
    return f2*np.exp(-0.25*(d-x)*(d-x)/(s*s))+f1*np.exp(-0.25*x*x/(s*s))

def pred_dlogL(d, f1, f2, s, bg):
    ''' Delta log likelihood between true two-star model and best fit one-star model '''
    f0 = bestfitflux(d, f1, f2, s)
    x = bestfitpos(d, f1, f2, s)
    return 0.5 * (f0*f0 - 2*np.exp(-0.25*x*x/(s*s))*f0*f1 +f1*f1 - 2*np.exp(-0.25*(d-x)*(d-x)/(s*s))*f0*f2 \
                 + 2*np.exp(-0.25*(d*d)/(s*s))*f1*f2 + f2*f2) / (4*np.pi*s*s*bg)



def predict_blended_catalog(true_xs, true_ys, true_fs, rmax=3.0, beam_fwhm=3., varnoise=1., alph=1.0, dlogprior=None):
    ''' 
    For a given truth catalog, we can iteratively 
        - choose the brightest source
        - look for any sources above Smin that are within some distance of the source
        - if there are several, choose the brightest neighbor. if not then add source to "convolved" catalog
        - compute the dlogL for the pair of sources, if it is above the parsimony prior threshold, then they can be deblended
        - if they can be deblended, add individual true fluxes to "convolved" catalog. if not, add best fit one-star flux to convolved catalog

    '''
    
    idx_list = np.argsort(true_fs)
    
    # we will start with brightest sources and work our way down
    ordered_fluxes = list(true_fs[idx_list])
    ordered_xs = list(true_xs[idx_list])
    ordered_ys = list(true_ys[idx_list])

    
    blended_cat_xs, blended_cat_ys, blended_cat_fs = [], [], []
    
    # algorithm goes through full catalog until no sources are left
    while len(ordered_fluxes) > 0:
        f1 = ordered_fluxes[-1]
        
        # find nearest neighbors within rmax
        matching_radii = np.sqrt((ordered_xs-ordered_xs[-1])**2 + (ordered_ys-ordered_ys[-1])**2)
        nearest_neighbors = np.where((matching_radii > 0)*(matching_radii < rmax))[0]
        
        # if no neighbors, assume source can be properly recovered
        if len(nearest_neighbors)==0:            
            blended_cat_fs.append(f1)
            blended_cat_xs.append(ordered_xs[-1])
            blended_cat_ys.append(ordered_ys[-1])
            del ordered_xs[-1]
            del ordered_ys[-1]
            del ordered_fluxes[-1]
            continue
            
            
        # find neighbor with largest flux density
        fluxes_within_rmax = np.array(ordered_fluxes)[nearest_neighbors]
        matching_radii_within_rmax = matching_radii[nearest_neighbors]
        max_neighbor = np.argmax(np.array(ordered_fluxes)[nearest_neighbors])
        max_neighbor_ordered_idx = nearest_neighbors[max_neighbor]
        
        f2 = ordered_fluxes[max_neighbor_ordered_idx]
        assert f1 != f2
        
        beam_sigma = beam_fwhm/2.355
        
        # compute difference in log-likelihood between two source model and one source model
        dist = matching_radii_within_rmax[max_neighbor]
        dlogL = pred_dlogL(dist, f1, f2, beam_sigma, varnoise)

        dlogL -= alph*1.5 # parsimony prior for one band scaled by alph hyperparameter
        
        # subtract any additional prior terms, not being used currently
        if dlogprior is not None:
            dlogL -= dlogprior 
        
        # compute the probability of it being a two-source configuration using posterior odds
        pn2 = 1./(1+np.exp(-dlogL))
        # draw on this probablility to determine whether source is simulated to be blended
        blend = (np.random.uniform(0, 1) > pn2)
        
        # if its a blend, take the best fit flux from the one source model, add it to "recovered" catalog, 
        # then remove neighbor and original source
        if blend:
            bestfit_f = bestfitflux(matching_radii_within_rmax[max_neighbor], f1, f2, beam_sigma)
            blended_cat_fs.append(bestfit_f)
            blended_cat_xs.append(ordered_xs[max_neighbor])
            blended_cat_ys.append(ordered_ys[max_neighbor])

            assert ordered_fluxes[max_neighbor_ordered_idx]==f2
            
            del ordered_xs[max_neighbor_ordered_idx]
            del ordered_ys[max_neighbor_ordered_idx]
            del ordered_fluxes[max_neighbor_ordered_idx]

        else:
            blended_cat_fs.append(f1)
            blended_cat_xs.append(ordered_xs[-1])
            blended_cat_ys.append(ordered_ys[-1])
            
        del ordered_xs[-1]
        del ordered_ys[-1]
        del ordered_fluxes[-1]
    
    return blended_cat_xs, blended_cat_ys, blended_cat_fs



    