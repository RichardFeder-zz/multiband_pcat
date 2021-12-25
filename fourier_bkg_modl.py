import numpy as np
import matplotlib 
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from image_eval import psf_poly_fit, image_model_eval
from scipy.ndimage import gaussian_filter
from spire_plotting_fns import plot_mp_fit


def compute_marginalized_templates(n_terms, data, error, imsz=None, bt_siginv_b=None, bt_siginv_b_inv=None,\
                           ravel_temps=None, fourier_templates=None,  psf_fwhm=3., \
                          ridge_fac = None, show=False, x_max_pivot=None, verbose=True):
    
    '''
    NOTE -- this only works for single band at the moment. Is there a way to compute the Moore-Penrose inverse for 
    backgrounds observed over several bands with a fixed color prior? 

    also , I think that using the full noise model in the matrix product is necessary when using multiband and multi-region
    evaluations. This might already be handled in the code by zeroing out NaNs.
    
    Condition number is not sensitive to multiplicative normalization tried this
    
    '''
    
    if imsz is None:
        imsz = error.shape
        
    if verbose:
        print('n_terms is ', n_terms)
        
    if ravel_temps is None:
        if fourier_templates is None:
            fourier_templates = make_fourier_templates(imsz[0], imsz[1], n_terms, psf_fwhm=psf_fwhm, x_max_pivot=x_max_pivot)
        ravel_temps = ravel_temps_from_ndtemp(fourier_templates, n_terms)
    
    im_cut_rav = data.copy().ravel()
    err_cut_rav = error.copy().ravel()
    
    # compress system of equations excluding any nan-valued entries
    nanmask = np.logical_or((err_cut_rav ==0), np.logical_or(np.isnan(err_cut_rav), np.isnan(im_cut_rav)))
    ravel_temps = ravel_temps.compress(~nanmask, axis=1)
    im_cut_rav = im_cut_rav.compress(~nanmask)
    err_cut_rav = err_cut_rav.compress(~nanmask)

    if bt_siginv_b_inv is None:
        if verbose:
            print('min max err cut rav:', np.min(err_cut_rav), np.max(err_cut_rav))
            print('min max err cut rav inv2:', np.min(np.diag(err_cut_rav**(-2))), np.max(np.diag(err_cut_rav**(-2))))
            print('min max ravel temps:', np.min(ravel_temps), np.max(ravel_temps))

        bt_siginv_b = np.dot(ravel_temps, np.dot(np.diag(err_cut_rav**(-2)), ravel_temps.transpose()))
            
        assert ~np.isnan(np.linalg.cond(bt_siginv_b))

        if verbose:
            print('condition number of (B^T S^{-1} B)^{-1}: ', np.linalg.cond(bt_siginv_b))
        
        if ridge_fac is not None:
            if verbose:
                print('adding regularization')
                
            lambda_I = np.zeros_like(bt_siginv_b)
            np.fill_diagonal(lambda_I, ridge_fac)
            bt_siginv_b += lambda_I
            
        bt_siginv_b_inv = np.linalg.inv(bt_siginv_b)
    
    siginv_K_rav = im_cut_rav*err_cut_rav**(-2) # Sigma^-1 Y
    bt_siginv_K = np.dot(ravel_temps, siginv_K_rav) # B^T Sigma^-1 Y
    A_hat = np.dot(bt_siginv_b_inv, bt_siginv_K) # (B^T Sigma^-1 B^-1 + Lambda I) B^T Sigma^-1 Y
    mp_coeffs = np.reshape(A_hat, (n_terms, n_terms, 4))
    temp_A_hat = generate_template(mp_coeffs, n_terms, fourier_templates=fourier_templates, N=imsz[0], M=imsz[1], x_max_pivot=x_max_pivot)

    if show:
        plot_mp_fit(temp_A_hat, n_terms, A_hat, data)
        
        
    return fourier_templates, ravel_temps, bt_siginv_b, bt_siginv_b_inv, mp_coeffs, temp_A_hat


def compute_Ahat_templates(n_terms, error, imsz=None, bt_siginv_b=None, bt_siginv_b_inv=None,\
                           ravel_temps=None, fourier_templates=None, data=None, psf_fwhm=3., \
                          mean_sig=True, ridge_fac = None, show=False, inpaint_nans=True, x_max_pivot=None):
    
    # NOTE -- this only works for single band at the moment. Is there a way to compute the Moore-Penrose inverse for 
    # backgrounds observed over several bands with a fixed color prior? 

    # also , I think that using the full noise model in the matrix product is necessary when using multiband and multi-region
    # evaluations. This migth already be handled in the code by zeroing out NaNs.

    if imsz is None:
        imsz = error.shape

    if ravel_temps is None:
        if fourier_templates is None:
            fourier_templates = make_fourier_templates(imsz[0], imsz[1], n_terms, psf_fwhm=psf_fwhm, x_max_pivot=x_max_pivot)
        ravel_temps = ravel_temps_from_ndtemp(fourier_templates, n_terms)
    
    err_cut_rav = error.ravel()

    if bt_siginv_b_inv is None:
        if mean_sig:
            bt_siginv_b = np.dot(ravel_temps, ravel_temps.transpose())
        else:
            bt_siginv_b = np.dot(ravel_temps, np.dot(np.diag(err_cut_rav**(-2)), ravel_temps.transpose()))
        
        print('condition number of (B^T S^{-1} B)^{-1}: ', np.linalg.cond(bt_siginv_b))
        
        if ridge_fac is not None:

            print('adding regularization')
            lambda_I = np.zeros_like(bt_siginv_b)
            np.fill_diagonal(lambda_I, ridge_fac)
            bt_siginv_b_inv = np.linalg.inv(bt_siginv_b + lambda_I)

        else:
            bt_siginv_b_inv = np.linalg.inv(bt_siginv_b)

    if mean_sig:
        bt_siginv_b_inv *= np.nanmean(error.astype(np.float64))**2
                
    
    if data is not None:
        im_cut_rav = data.ravel()

        if inpaint_nans:
            nan_idxs = np.where(np.isnan(im_cut_rav))[0]
            for nan_idx in nan_idxs:
                im_cut_rav[nan_idx] = np.nanmean(im_cut_rav)

        if mean_sig:
            siginv_K_rav = im_cut_rav*np.nanmean(error)**(-2)
        else:
            siginv_K_rav = im_cut_rav*err_cut_rav**(-2)

        siginv_K_rav[np.isinf(siginv_K_rav)] = 0.
        bt_siginv_K = np.dot(ravel_temps, siginv_K_rav)
        A_hat = np.dot(bt_siginv_b_inv, bt_siginv_K)


        mp_coeffs = np.reshape(A_hat, (n_terms, n_terms, 4))

        temp_A_hat = generate_template(mp_coeffs, n_terms, fourier_templates=fourier_templates, N=imsz[0], M=imsz[1], x_max_pivot=x_max_pivot)

        if show:
            plot_mp_fit(temp_A_hat, n_terms, A_hat, data)

        
        return fourier_templates, ravel_temps, bt_siginv_b, bt_siginv_b_inv, mp_coeffs
    
    return fourier_templates, ravel_temps, bt_siginv_b_inv, None


def plot_mp_fit(temp_A_hat, n_terms, A_hat, data):
    plt.figure(figsize=(10,10))
    plt.suptitle('Moore-Penrose inverse, $N_{FC}$='+str(n_terms), fontsize=20)
    plt.subplot(2,2,1)
    plt.title('Background estimate', fontsize=18)
    plt.imshow(temp_A_hat, origin='lower', cmap='Greys', vmax=np.percentile(temp_A_hat, 99), vmin=np.percentile(temp_A_hat, 1))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(2,2,2)
    plt.hist(np.abs(A_hat), bins=np.logspace(-5, 1, 30))
    plt.xscale('log')
    plt.xlabel('Absolute value of Fourier coefficients', fontsize=14)
    plt.ylabel('N')
    plt.subplot(2,2,3)
    plt.title('Image', fontsize=18)
    plt.imshow(data, origin='lower', cmap='Greys', vmax=np.percentile(temp_A_hat, 95), vmin=np.percentile(temp_A_hat, 5))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(2,2,4)
    plt.title('Image - Background estimate', fontsize=18)
    plt.imshow(data-temp_A_hat, origin='lower', cmap='Greys', vmax=np.percentile(temp_A_hat, 95), vmin=np.percentile(temp_A_hat, 5))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def ravel_temps_from_ndtemp(templates, n_terms, auxdim=4):
    ''' Faster version of previous function '''
    ravel_templates_all = np.reshape(templates, (templates.shape[0], templates.shape[1], auxdim, templates.shape[-2]*templates.shape[-1]))
    
    ravel_temps = ravel_templates_all[:n_terms, :n_terms]
    
    ravel_temps = np.reshape(ravel_temps, (ravel_temps.shape[0]*ravel_temps.shape[1]*ravel_temps.shape[2], ravel_temps.shape[-1]))
    
    return ravel_temps

def multiband_fourier_templates(imszs, n_terms, show_templates=False, psf_fwhms=None, x_max_pivot_list=None):
    '''
    Given a list of image and beam sizes, produces multiband fourier templates for background modeling.

    Parameters
    ----------

    imszs : list of lists
        List containing image dimensions for each of the three observations

    n_terms : int
        Order of Fourier expansion for templates. the number of templates (currently) scales as 2*n_terms^2

    show_templates : bool, optional
        if True, plots the array of templates. Default is False.

    psf_fwhms : list, optional
        List of beam sizes across observations. If left unspecified, all PSFs assumed to have 3 pixel FWHM. 
        Default is 'None'.
    
    Returns
    -------

    all_templates : list of `numpy.ndarray's
        The set of Fourier templates for each observation.

    '''

    all_templates = []

    for b in range(len(imszs)):
        if psf_fwhms is None:
            psf_fwhm = None
        else:
            psf_fwhm = psf_fwhms[b]

        x_max_pivot = None
        if x_max_pivot_list is not None:
            x_max_pivot = x_max_pivot_list[b]

        all_templates.append(make_fourier_templates(imszs[b][0], imszs[b][1], n_terms, show_templates=show_templates, psf_fwhm=psf_fwhm, x_max_pivot=x_max_pivot))
    return all_templates

def make_fourier_templates(N, M, n_terms, show_templates=False, psf_fwhm=None, shift=False, x_max_pivot=None):
        
    '''
    
    Given image dimensions and order of the series expansion, generates a set of 2D fourier templates.

    Parameters
    ----------

    N : int
        length of image
   
    M : int
        width of image
    
    n_terms : int
        Order of Fourier expansion for templates. the number of templates (currently) scales as 2*n_terms^2
    
    show_templates : bool, optional
        if True, plots the array of templates. Default is False.
    
    psf_fwhm : float, optional
        Observation PSF full width at half maximum (FWHM). This can be used to pre-convolve templates for background modeling 
        Default is 'None'.

    x_max_pivot : float, optional
        Indicating pixel coordinate for boundary of FOV in each dimension. Default is 'None'.

    Returns
    -------
    
    templates : `numpy.ndarray' of shape (n_terms, n_terms, 4, N, M)
        Contains 2D Fourier templates for truncated series


    '''

    templates = np.zeros((n_terms, n_terms, 4, N, M))

    x = np.arange(N)
    y = np.arange(M)
    
    meshx, meshy = np.meshgrid(x, y)
        
    xtemps_cos = np.zeros((n_terms, N, M))
    ytemps_cos = np.zeros((n_terms, N, M))
    xtemps_sin = np.zeros((n_terms, N, M))
    ytemps_sin = np.zeros((n_terms, N, M))

    N_denom = N
    M_denom = M

    if x_max_pivot is not None:
        N_denom = x_max_pivot
        M_denom = x_max_pivot

    for n in range(n_terms):

        # modified series
        if shift:
            xtemps_sin[n] = np.sin((n+1-0.5)*np.pi*meshx/N_denom)
            ytemps_sin[n] = np.sin((n+1-0.5)*np.pi*meshy/M_denom)
        else:
            xtemps_sin[n] = np.sin((n+1)*np.pi*meshx/N_denom)
            ytemps_sin[n] = np.sin((n+1)*np.pi*meshy/M_denom)
        
        xtemps_cos[n] = np.cos((n+1)*np.pi*meshx/N_denom)
        ytemps_cos[n] = np.cos((n+1)*np.pi*meshy/M_denom)
    
    for i in range(n_terms):
        for j in range(n_terms):

            if psf_fwhm is not None: # if beam size given, convolve with PSF assumed to be Gaussian
                templates[i,j,0,:,:] = gaussian_filter(xtemps_sin[i]*ytemps_sin[j], sigma=psf_fwhm/2.355)
                templates[i,j,1,:,:] = gaussian_filter(xtemps_sin[i]*ytemps_cos[j], sigma=psf_fwhm/2.355)
                templates[i,j,2,:,:] = gaussian_filter(xtemps_cos[i]*ytemps_sin[j], sigma=psf_fwhm/2.355)
                templates[i,j,3,:,:] = gaussian_filter(xtemps_cos[i]*ytemps_cos[j], sigma=psf_fwhm/2.355)

            else:
                templates[i,j,0,:,:] = xtemps_sin[i]*ytemps_sin[j]
                templates[i,j,1,:,:] = xtemps_sin[i]*ytemps_cos[j]
                templates[i,j,2,:,:] = xtemps_cos[i]*ytemps_sin[j]
                templates[i,j,3,:,:] = xtemps_cos[i]*ytemps_cos[j]
     
    if show_templates:
        for k in range(4):
            counter = 1
            plt.figure(figsize=(8,8))
            for i in range(n_terms):
                for j in range(n_terms):           
                    plt.subplot(n_terms, n_terms, counter)
                    plt.title('i = '+ str(i)+', j = '+str(j))
                    plt.imshow(templates[i,j,k,:,:])
                    counter +=1
            plt.tight_layout()
            plt.show()

    return templates


def generate_template(fourier_coeffs, n_terms, fourier_templates=None, imsz=None, N=None, M=None, psf_fwhm=None, x_max_pivot=None):

    '''
    Given a set of coefficients and Fourier templates, computes their dot product.

    Parameters
    ----------

    fourier_coeffs : `~numpy.ndarray' of shape (n_terms, n_terms, 2)
        Coefficients of truncated Fourier expansion.

    n_terms : int
        Order of Fourier expansion to compute sum over. This is left explicit as an input
        in case one wants the flexibility of calling it for different numbers of terms, even
        if the underlying truncated series has more terms.

    fourier_templates : `~numpy.ndarray' of shape (n_terms, n_terms, 2, N, M), optional
        Contains 2D Fourier templates for truncated series. If left unspecified, a set of Fourier templates is generated
        on the fly. Default is 'None'.

    N : int, optional
        length of image. Default is 'None'.
   
    M : int
        width of image. Default is 'None.'

    psf_fwhm : float, optional
        Observation PSF full width at half maximum (FWHM). This can be used to pre-convolve templates for background modeling 
        Default is 'None'.

    x_max_pivot : float, optional
        Because of different image resolution across bands and the use of multiple region proposals, the non pivot band images may cover a larger 
        field of view than the pivot band image. When modeling structured emission across several bands, it is important that the Fourier components
        model a consistent field of view. Extra pixels in the non-pivot bands do not contribute to the log-likelihood, so I think the solution is to 
        compute the Fourier templates where the period is based on the WCS transformations across bands, which can translate coordinates bounding
        the pivot image to coordinates in the non-pivot band images.

        Default is 'None'. 

    Returns
    -------

    sum_temp : `~numpy.ndarray' of shape (N, M)
        The summed template.

    '''

    if imsz is None and N is None and M is None:
        print('need to provide input dimensions through either imsz or (N, M)')
        return None

    if imsz is None:
        imsz = [N,M]
        
    if fourier_templates is None:
        fourier_templates = make_fourier_templates(imsz[0], imsz[1], n_terms, psf_fwhm=psf_fwhm, x_max_pivot=x_max_pivot)

    sum_temp = np.sum([fourier_coeffs[i,j,k]*fourier_templates[i,j,k] for i in range(n_terms) for j in range(n_terms) for k in range(fourier_coeffs.shape[-1])], axis=0)
    
    return sum_temp

def fit_coeffs_to_observed_comb(observed_comb, obs_noise_sig,ftemplates, true_fcoeffs = None, true_comb=None, n_terms=None, sig_dtemp=0.1, niter=100, init_nsig=1.):
    if true_fcoeffs is not None:
        init_fcoeffs = np.random.normal(0, obs_noise_sig, size=(true_fcoeffs.shape[0], true_fcoeffs.shape[1], 4))

        n_terms = init_fcoeffs.shape[0]

    elif n_terms is not None:
        init_fcoeffs = np.random.normal(0, obs_noise_sig, size=(n_terms, n_terms, 4))

            
    running_fcoeffs = init_fcoeffs.copy()
    
    all_running_fcoeffs = np.zeros((niter//1000, n_terms, n_terms, 4))

    temper_schedule = np.logspace(np.log10(init_nsig), np.log10(1.), niter)
    print('temper schedule: ', temper_schedule)
    print(init_fcoeffs.shape, n_terms, ftemplates.shape)
    
    lazy_temp = generate_template(init_fcoeffs, n_terms, ftemplates)
    running_temp = lazy_temp.copy()
    lnLs = np.zeros((niter,))
    lnL = -0.5*np.sum((1./obs_noise_sig**2)*(observed_comb - running_temp)*(observed_comb-running_temp))
    lnLs[0] = lnL
    accepts= np.zeros((niter,))
    
    perts = np.random.normal(0, sig_dtemp, niter)
    
    nsamp = 0
    for n in range(niter):
        
        sig_dtemp_it = temper_schedule[n]*sig_dtemp
        
        idxk = np.random.randint(0, 2)
        idx0, idx1 = np.random.randint(0, n_terms), np.random.randint(0, n_terms)

        prop_dtemp = ftemplates[idx0,idx1,idxk,:,:]*perts[n]
        plogL = -0.5*np.sum((1./obs_noise_sig**2)*(observed_comb - running_temp - prop_dtemp)*(observed_comb-running_temp - prop_dtemp))
        
        dlogP = plogL - lnL
        
        accept_or_not = (np.log(np.random.uniform()) < dlogP)
        accepts[n] = int(accept_or_not)
        if accept_or_not:
            running_temp += prop_dtemp
            running_fcoeffs[idx0, idx1, idxk] += perts[n]
            lnLs[n] = plogL
            lnL = plogL
        else:
            lnLs[n] = lnL
        
        if n%5000==0:
            print('n = ', n)
            
        if n%1000==0:
            all_running_fcoeffs[nsamp,:,:,:] = running_fcoeffs
            nsamp += 1
            
        if n%(niter//10)==0:

            plt.figure(figsize=(16, 4))
            plt.suptitle('n = '+str(n), fontsize=20, y=1.02)
            plt.subplot(1,4,3)
            plt.title('model', fontsize=16)
            plt.imshow(running_temp)
            plt.colorbar()
            plt.subplot(1,4,2)
            plt.title('observed', fontsize=16)
            plt.imshow(observed_comb)
            plt.colorbar()
            plt.subplot(1,4,1)
            if true_comb is not None:
                plt.title('truth')
                plt.imshow(true_comb - np.mean(true_comb))
            else:
                plt.title('observed - model')
                plt.imshow(observed_comb-running_temp)
            plt.colorbar()
            plt.subplot(1,4,4)
            plt.title('$\\delta b(x,y)/\\sigma(x,y)$', fontsize=16)

            if true_comb is not None:
                resid = (observed_comb - running_temp)/obs_noise_sig
                plt.imshow(resid, vmin=np.percentile(resid, 5), vmax=np.percentile(resid, 95))
                plt.colorbar()
            plt.tight_layout()
            plt.show()
    
    print(np.mean(accepts))

def plot_logL(lnlz, N=100, M=100):

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(lnlz)), -2*lnlz, label='Chain min $\\chi_{red.}^2 = $'+str(np.round(np.min(-2*lnlz)/(N*M), 2)))
    plt.axhline(N*M, linestyle='dashed', label='$\\chi_{red.}^2 = 1$')
    plt.legend(fontsize=14)
    plt.yscale('log')
    plt.ylabel('$-2\\ln\\mathcal{L}$', fontsize=18)
    plt.xlabel('Sample iteration', fontsize=18)
    plt.tight_layout()
    plt.show() 


def compute_BT_B(n_terms, M, auxdim=4):
    # this is supposed to be for computing the analytic covariance matrix of the fourier components, don't think it works, don't bother
    bt_b = np.zeros((auxdim*n_terms**2, auxdim*n_terms**2))
    
    for i_d in range(n_terms):
        for j_d in range(n_terms):
            for k_d in range(auxdim):
                
                for i in range(n_terms):
                    for j in range(n_terms):
                        for k in range(auxdim):

                            term1 = 0.
                            term2 = 0.

                            if i==i_d:
                                
                                if (k==0 and k_d==1) or (k==1 and k_d==0) or (k==2 and k_d==3) or (k==3 and k_d==2) or (k==0 and k_d==0) or (k==1 and k_d==1) or (k==2 and k_d==2) or (k==3 and k_d==3):
                                    term1 = M/2.
                        
                            else:
                                if (k==0 and k_d==2) or (k==0 and k_d==3) or (k==1 and k_d==2) or (k==1 and k_d==3) or (k==2 and k_d==0) or (k==2 and k_d==1) or (k==3 and k_d==0) or (k==3 and k_d==1):
                                    
                                    term1 = (M/np.pi)*((1/(2*(i-i_d)))*(1-np.cos(np.pi*(i-i_d))) +(1/(2*(i+i_d+2.)))*(1-np.cos(np.pi*(i+i_d))))
                                                                                                        
                            if j==j_d:
                                
                                if (k==0 and k_d==2) or (k==1 and k_d==3) or (k==2 and k_d==0) or (k==3 and k_d==1):
                                    term2 = M/2.
                                elif (k==0 and k_d==0) or (k==1 and k_d==1) or (k==2 and k_d==2) or (k==3 and k_d==3):
                                    term2 = M/2.
                                
                            else:
                                if (k==0 and k_d==1) or (k==0 and k_d==3) or (k==1 and k_d==0) or (k==1 and k_d==2) or (k==2 and k_d==1) or (k==2 and k_d==3) or (k==3 and k_d==0) or (k==3 and k_d==2):                         
                                    
                                    term2 = (M/np.pi)*((1./(2.*(j-j_d)))*(1.-np.cos(np.pi*(j-j_d))) +(1./(2.*(j+j_d+2.)))*(1.-np.cos(np.pi*(j+j_d))))
                                
                                
                            bt_b[i_d*j_d*auxdim + j_d*auxdim + k_d, i*j*auxdim + j*auxdim + k] = term1*term2
            
    return bt_b


