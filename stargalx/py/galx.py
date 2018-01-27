import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='poster', style='ticks', color_codes=True)
import sys, os, h5py
import scipy as sp

def summgene(varb):
    print np.amin(varb)
    print np.amax(varb)
    print np.mean(varb)
    print varb.shape
    print


def retr_factsers(sersindx):
    
    factsers = 1.9992 * sersindx - 0.3271

    return factsers


def retr_sers(numbsidegrid=20, sersindx=4., factusam=100, factradigalx=2., pathplot=None):
    
    # numbsidegrid is the number of pixels along one side of the grid and should be an odd number

    numbsidegridusam = numbsidegrid * factusam

    # generate the grid
    xpostemp = np.linspace(-factradigalx, factradigalx, numbsidegridusam + 1)
    ypostemp = np.linspace(-factradigalx, factradigalx, numbsidegridusam + 1)
    xpos, ypos = np.meshgrid(xpostemp, ypostemp)
    
    # evaluate the Sersic profile
    ## approximation to the b_n factor
    factsers = retr_factsers(sersindx)
    ## profile sampled over the square grid
    amplphonusam = np.exp(-factsers * (np.sqrt(xpos**2 + ypos**2)**(1. / sersindx) - 1.))#.flatten()
    ## down sample
    amplphon = np.empty((numbsidegrid + 1, numbsidegrid + 1))
    for k in range(numbsidegrid + 1):
        for l in range(numbsidegrid + 1):
            amplphon[k, l] = np.mean(amplphonusam[k*factusam:(k+1)*factusam, l*factusam:(l+1)*factusam])
    #indx = np.arange(factusam / 2, numbsidegridusam - factusam / 2 + 1, factusam) 
    indx = np.linspace(factusam / 2, numbsidegridusam - factusam / 2 + 1, numbsidegrid + 1, dtype=int) 
    
    xpostemp = xpostemp[indx]
    ypostemp = ypostemp[indx]
    
    xpos, ypos = np.meshgrid(xpostemp, ypostemp)
    
    amplphon = amplphon.flatten()
    ## normalize such that the sum is unity
    amplphon /= sum(amplphon)
    
    gridphon = np.vstack((xpos.flatten(), ypos.flatten()))
    
    if pathplot != None:
        figr, axis = plt.subplots(figsize=(6, 6))
        radi = np.sqrt(xpos**2 + ypos**2).flatten()
        axis.plot(radi, amplphon, ls='', marker='o', markersize=1)
        axis.set_yscale('log')
        axis.set_xlabel(r'$\theta$ [pixel]')
        axis.set_ylabel(r'$\Sigma$')
        axis.axvline(1., ls='--')
        path = pathplot + 'sersprof%03.2g.pdf' % sersindx
        plt.tight_layout()
        plt.savefig(path)
        plt.close(figr)

    return gridphon, amplphon


def main():
    
    # sample the Sersic profile to get the phonion positions and amplitudes

    return gridphon, amplphon


def samp_powr(minm, maxm, indx, size=1):
    
    cdfn = np.random.random(size=size)
    minmtran = minm**(1. - indx)
    maxmtran = maxm**(1. - indx)

    icdf = (minmtran + (maxmtran - minmtran) * cdfn)**(1. / (1. - indx))
    
    return icdf


def retr_tranphon(gridphon, amplphon, xposgalx, yposgalx, specgalx, argsfrst, argsseco, argsthrd, paratype='fris'):
    
    # find the transformation matrix if parametrization is 
    if paratype == 'fris':
        ## orientation vector of a frisbee
        avec = np.array([argsfrst, argsseco, argsthrd])
        normavec = np.sqrt(sum(avec**2))
        avecunit = avec / normavec
        bvec = np.array([argsseco, -argsfrst, 0.])
        normbvec = np.sqrt(sum(bvec**2))
        bvecunit = bvec / normbvec
        cvec = np.cross(avecunit, bvecunit)
        tranmatr = normavec * np.array([[argsseco / normbvec, cvec[0]], [-argsfrst / normbvec, cvec[1]]])
        cvecnorm = np.sqrt(argsfrst**2 * argsthrd**2 + argsseco**2 * argsthrd**2 + (argsfrst**2 + argsseco**2)**2)
        tranmatr = normavec * np.array([[argsseco / normbvec, argsfrst * argsthrd / cvecnorm], [-argsfrst / normbvec, argsseco * argsthrd / cvecnorm]])
        radigalx = normavec
    if paratype == 'sher':
        ## size, axis ratio, and orientation angle
        radigalx = argsfrst 
        shexgalx = argsseco 
        sheygalx = argsthrd 
        tranmatr = radigalx * np.array([[1., -shexgalx], [-sheygalx, 1.]])
    elif paratype == 'angl':
        ## size, horizontal and vertical shear
        radigalx = argsfrst 
        ratigalx = argsseco 
        anglgalx = argsthrd 
        tranmatr = radigalx * np.array([[ratigalx * np.cos(anglgalx), -np.sin(anglgalx)], [ratigalx * np.sin(anglgalx), np.cos(anglgalx)]])

    # transform the phonion grid
    gridphontran = np.matmul(tranmatr, gridphon)

    # positions of the phonions
    xposphon = xposgalx + gridphontran[0, :]
    yposphon = yposgalx + gridphontran[1, :]
    
    # spectrum of the phonions
    specphon = amplphon[None, :] * specgalx[:, None]
    
    return xposphon, yposphon, specphon


def retr_sbrt(spec, radigalx, sersindx):
    
    factsers = retr_factsers(sersindx)
   
    # temp -- only works for sersindx == 4
    sbrt = spec / 7.2 / np.pi / radigalx[None, :]**2
    
    #fact = 1. / 2. / np.pi * factsers**(sersindx*2) / sersindx / sp.special.gamma(2. * sersindx)
    #print 'retr_sbrt'
    #print 'sersindx'
    #print sersindx
    #print 'fact'
    #print fact
    #print
    #sbrt = fact * spec / radigalx**2
    
    return sbrt


def retr_sizephonmrkr(specphon):
    
    # minimum and maximum marker sizes
    minm = 1e-4
    maxm = 1e6
    size = 5. * (np.log10(specphon) - np.log10(minm)) / (np.log10(maxm) - np.log10(minm))
    if ((specphon < minm) | (specphon > maxm)).any():
        print 'specphon'
        summgene(specphon)
        raise

    return size


def writ_truedata():
    
    pathlion = os.environ["LION_PATH"]
    sys.path.insert(0, pathlion)
    from image_eval import image_model_eval, psf_poly_fit
    
    dictglob = dict()
    
    sersindx = 2.
    numbsidegrid = 20

    pathliondata = os.environ["LION_DATA_PATH"] + '/data/'
    pathlionimag = os.environ["LION_DATA_PATH"] + '/imag/'
    fileobjt = open(pathliondata + 'sdss.0921_psf.txt')
    numbsidepsfn, factsamp = [np.int32(i) for i in fileobjt.readline().split()]
    fileobjt.close()
    
    psfn = np.loadtxt(pathliondata + 'sdss.0921_psf.txt', skiprows=1).astype(np.float32)
    cpsf = psf_poly_fit(psfn, factsamp)
    cpsf = cpsf.reshape((-1, cpsf.shape[2]))
    
    np.random.seed(0)
    numbside = [100, 100]
    
    # generate stars
    numbstar = 1
    fluxdistslop = np.float32(2.0)
    minmflux = np.float32(250.)
    logtflux = np.random.exponential(scale=1. / (fluxdistslop - 1.), size=numbstar).astype(np.float32)
    
    dictglob['numbstar'] = numbstar
    dictglob['xposstar'] = (np.random.uniform(size=numbstar) * (numbside[0] - 1)).astype(np.float32)
    dictglob['yposstar'] = (np.random.uniform(size=numbstar) * (numbside[1] - 1)).astype(np.float32)
    dictglob['fluxdistslop'] = fluxdistslop
    dictglob['minmflux'] = minmflux
    dictglob['fluxstar'] = minmflux * np.exp(logtflux)
    dictglob['specstar'] = dictglob['fluxstar'][None, :]
    
    dictglob['back'] = np.float32(179.)
    dictglob['gain'] = np.float32(4.62)
    
    # generate galaxies
    numbgalx = 100
    dictglob['numbgalx'] = numbgalx
    
    dictglob['xposgalx'] = (np.random.uniform(size=numbgalx) * (numbside[0] - 1)).astype(np.float32)
    dictglob['yposgalx'] = (np.random.uniform(size=numbgalx) * (numbside[1] - 1)).astype(np.float32)

    dictglob['sizegalx'] = samp_powr(2., 10., 1.5, size=numbgalx)
    dictglob['avecfrst'] = (2. * np.random.uniform(size=numbgalx) - 1.).astype(np.float32)
    dictglob['avecseco'] = (2. * np.random.uniform(size=numbgalx) - 1.).astype(np.float32)
    dictglob['avecthrd'] = (2. * np.random.uniform(size=numbgalx) - 1.).astype(np.float32)
    mgtd = np.sqrt(dictglob['avecfrst']**2 + dictglob['avecseco']**2 + dictglob['avecthrd']**2)
    dictglob['avecfrst'] /= mgtd
    dictglob['avecseco'] /= mgtd
    dictglob['avecthrd'] /= mgtd
    dictglob['specgalx'] = samp_powr(2500., 25000., 2., size=numbgalx)[None, :]
    dictglob['sbrtgalx'] = retr_sbrt(dictglob['specgalx'], dictglob['sizegalx'], sersindx)
    
    gdat = gdatstrt()
    
    listsersindx = [0.5, 1., 2., 4., 6., 8., 10.]
    for sersindx in listsersindx:
        gdat.sersindx = sersindx
        gridphon, amplphon = retr_sers(numbsidegrid=numbsidegrid, sersindx=sersindx, pathplot=pathlionimag)
    
        gridphon, amplphon = retr_sers(numbsidegrid=numbsidegrid, sersindx=sersindx, pathplot=pathlionimag)
        numbphongalx = (numbsidegrid + 1)**2
        numbener = 1
        numbphon = numbgalx * numbphongalx
        xposphon = np.empty(numbphon)
        yposphon = np.empty(numbphon)
        specphon = np.empty((numbener, numbphon))
        for k in range(dictglob['numbgalx']):
            indx = np.arange(k * numbphongalx, (k + 1) * numbphongalx)
            gridphontemp = gridphon * dictglob['sizegalx'][k]
            xposphon[indx], yposphon[indx], specphon[:, indx] = retr_tranphon(gridphontemp, amplphon, dictglob['xposgalx'][k], dictglob['yposgalx'][k], dictglob['specgalx'][:, k], \
                                                                                                 dictglob['avecfrst'][k], dictglob['avecseco'][k], dictglob['avecthrd'][k], 'fris')
        
    
        xposcand = np.concatenate((xposphon, dictglob['xposstar'])).astype(np.float32)
        yposcand = np.concatenate((yposphon, dictglob['yposstar'])).astype(np.float32)
        speccand = np.concatenate((specphon, dictglob['specstar']), axis=1).astype(np.float32)
        
        indx = np.where((xposcand < 100.) & (xposcand > 0.) & (yposcand < 100.) & (yposcand > 0.))[0]
        xposcand =  xposcand[indx]
        yposcand =  yposcand[indx]
        speccand =  speccand[:, indx]
        
        # generate data
        datacnts = image_model_eval(xposcand, yposcand, speccand[0, :], dictglob['back'], numbside, numbsidepsfn, cpsf)
        #datacnts[datacnts < 1] = 1. # maybe some negative pixels
        variance = datacnts / dictglob['gain']
        datacnts += (np.sqrt(variance) * np.random.normal(size=(numbside[1],numbside[0]))).astype(np.float32)
        dictglob['datacnts'] = datacnts 
        # auxiliary data
        dictglob['numbside'] = numbside
        dictglob['psfn'] = psfn
        
        # write data to disk
        path = pathliondata + 'true.h5'
        filearry = h5py.File(path, 'w')
        for attr, valu in dictglob.iteritems():
            filearry.create_dataset(attr, data=valu)
        filearry.close()
    
        gdat.pathlionimag = pathlionimag
        plot_cntsmaps(gdat, datacnts, 'truedatacnts')
        plot_cntsmaps(gdat, datacnts, 'truedatacntsphon', xposcand=xposcand, yposcand=yposcand, speccand=speccand)
  

class gdatstrt(object):
    
    def __init__(self):
        pass


def plot_cntsmaps(gdat, maps, name, xposcand=None, yposcand=None, speccand=None):
    
    figr, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(np.arcsinh(maps), interpolation='nearest')
    if speccand != None:
        size = retr_sizephonmrkr(speccand[0, :])
        axis.scatter(xposcand, yposcand, s=size, alpha=0.2)
    axis.set_ylim([0., 100.])
    axis.set_xlim([0., 100.])
    path = gdat.pathlionimag + name + '%03.2g.pdf' % gdat.sersindx
    plt.savefig(path)
    plt.close(figr)


def main():
    
    # sample the Sersic profile to get the phonion positions and amplitudes
    gridphon, amplphon = retr_sers()
    
    sizesqrt = 5. / np.sqrt(2.)
        
    # plot phonions whose positions have been stretched and rotated, and spectra rescaled
    if False:
        listparagalx = [ \
                        #[0.,   0.,  100.,        5.,        0.,              0., 'angl'], \
                        #[0.,  -2.,  100.,        5.,        0.,              0., 'angl'], \
                        #[-2., -2.,  100.,        5.,        0.,              0., 'angl'], \
                        #[-2.,  0.,  100.,        5.,        0.,              0., 'angl'], \
                        #
                        #[0.,   0.,  100.,        5.,        0.,              0., 'angl'], \
                        #[0.,   0.,  100.,       7.5,        0.,              0., 'angl'], \
                        #
                        #[0.,   0.,  100.,        5.,        0.,              0., 'angl'], \
                        #[0.,   0.,  500.,        5.,        0.,              0., 'angl'], \
                        #[0.,   0., 2500.,        5.,        0.,              0., 'angl'], \
                        #
                        #[0.,   0.,  100.,        5.,        1.,              0., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.2,              0., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5,              0., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5,      np.pi / 8., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5,      np.pi / 4., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5, 3. * np.pi / 8., 'angl'], \
                        #[0.,   0.,  100.,        5.,       1.5,      np.pi / 2., 'angl'], \
                        #
                        #[0.,   0.,  100.,        5.,        0.,              0., 'sher'], \
                        #[0.,   0.,  100.,        5.,        0.,             0.1, 'sher'], \
                        #[0.,   0.,  100.,        5.,        0.,             0.5, 'sher'], \
                        #[0.,   0.,  100.,        5.,        0.,              1., 'sher'], \
                        #[0.,   0.,  100.,        5.,        0.,              2., 'sher'], \
                        #[0.,   0.,  100.,        5.,       0.1,              0., 'sher'], \
                        #[0.,   0.,  100.,        5.,       0.5,              0., 'sher'], \
                        #[0.,   0.,  100.,        5.,        1.,              0., 'sher'], \
                        #[0.,   0.,  100.,        5.,        2.,              0., 'sher'], \
                        
                        [0.,   0.,  100.,        5.,        0.,              0., 'fris'], \
                        [0.,   0.,  100.,  sizesqrt,  sizesqrt,        sizesqrt, 'fris'], \
                        [0.,   0.,  100.,        5.,        5.,              5., 'fris'], \
                        [0.,   0.,  100.,        5.,        3.,              2., 'fris'], \
                        [0.,   0.,  100.,        5.,        1.,              2., 'fris'], \
                        [0.,   0.,  100.,        5.,        0.,              2., 'fris'], \
                        [0.,   0.,  100.,        5.,       -1.,              2., 'fris'], \
                       ] 
    else:
        listparagalx = []
        listavec = []
        numbiterfrst = 5
        numbiterseco = 2
        numbiterthrd = 2
        listavecfrst = np.linspace(-1., 1., numbiterfrst)
        listavecseco = np.linspace(0., 1., numbiterseco)
        listavecthrd = np.linspace(0., 1., numbiterthrd)
        for k in range(numbiterfrst):
            for l in range(numbiterseco):
                for m in range(numbiterthrd):
                    if listavecfrst[k] == 0. and listavecseco[l] == 0. and listavecthrd[m] == 0.:
                        continue
                    listparagalx.append([0., 0., np.array([10.]), listavecfrst[k], listavecseco[l], listavecthrd[m], 'fris'])
                    listavec.append([listavecfrst[k], listavecseco[l], listavecthrd[m]])
    
    plot_phon(listparagalx, gridphon, amplphon, listavec)


def plot_phon(listparagalx, gridphon, amplphon, listavec=None):
    
    # minimum flux allowed by the metamodel in ADU
    minmflux = 250

    pathplot = os.environ["LION_DATA_PATH"] + '/imag/'
    os.system('mkdir -p ' + pathplot)
    
    os.system('rm -rf ' + pathplot + '*')
    numbiter = len(listparagalx)

    for k in range(numbiter):
        
        xposgalx, yposgalx, specgalx, argsfrst, argsseco, argsthrd, paratype = listparagalx[k]
        figr, axis = plt.subplots(figsize=(6, 6))
        
        xposphon, yposphon, specphon = retr_tranphon(gridphon, amplphon, xposgalx, yposgalx, specgalx, argsfrst, argsseco, argsthrd, paratype)
        size = retr_sizephonmrkr(specphon[0, :])
        
        indxgrtr = np.where(specphon[0, :] > minmflux)
        axis.scatter(xposphon, yposphon, s=size, color='g')
        axis.scatter(xposphon[indxgrtr], yposphon[indxgrtr], s=size[indxgrtr], color='b')
        
        if listavec != None:
            axis.set_title('$A_x=%4.2g, A_y=%4.2g, A_z=%4.2g$' % (listavec[k][0], listavec[k][1], listavec[k][2]))
        axis.set_xlim([-4, 4])
        axis.set_ylim([-4, 4])
        axis.set_xlabel('$x$')
        axis.set_ylabel('$y$')
    
        path = pathplot + 'galx%s.pdf' % k
        plt.tight_layout()
        figr.savefig(path)
        plt.close(figr)


main()
#writ_truedata()
