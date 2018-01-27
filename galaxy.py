import numpy as np

def to_moments(r, theta, phi):
    u = np.cos(theta)
    xx = r*r*(1.+u*u+(1.-u*u)*np.cos(2.*phi))/4.
    xy = r*r*(1.-u*u)*np.sin(2.*phi)/4.
    yy = r*r*(1.+u*u-(1.-u*u)*np.cos(2.*phi))/4.
    if (xy*xy > xx*yy).any():
     mask = xy*xy > xx*yy
     print r[mask], theta[mask]/np.pi, phi[mask]/np.pi
     print np.cos(theta[mask])
     print xx[mask], xy[mask], yy[mask]
     print xx[mask]*yy[mask], xy[mask]*xy[mask]
     assert False
    return xx, xy, yy

def from_moments(xx, xy, yy):
    assert (xx*yy > xy*xy).all()
    a = np.sqrt(xx+yy+np.sqrt((xx-yy)*(xx-yy)+4*xy*xy))
    b = np.sqrt(xx+yy-np.sqrt((xx-yy)*(xx-yy)+4*xy*xy))
    theta = np.arccos(b/a)
    phi = np.arctan2(2*xy, -(yy-xx))/2.
    return a, theta, phi 

def retr_factsers(sersindx):
    factsers = 1.9992 * sersindx - 0.3271
    return factsers

def retr_sers(numbsidegrid=21, sersindx=4., factusam=101, factradigalx=2., pathplot=None):
    # numbsidegrid is the number of pixels along one side of the grid and should be an odd number
    numbsidegridusam = numbsidegrid * factusam

    # generate the grid
    xpostemp = np.linspace(-factradigalx, factradigalx, numbsidegridusam) #+1
    ypostemp = np.linspace(-factradigalx, factradigalx, numbsidegridusam) #+1
    xpos, ypos = np.meshgrid(xpostemp, ypostemp)

    # evaluate the Sersic profile
    ## approximation to the b_n factor
    factsers = retr_factsers(sersindx)
    ## profile sampled over the square grid
    amplphonusam = np.exp(-factsers * (np.sqrt(xpos**2 + ypos**2)**(1. / sersindx) - 1.))#.flatten()
    ## down sample
    amplphon = np.empty((numbsidegrid, numbsidegrid)) #+1
    for k in range(numbsidegrid): #+1
        for l in range(numbsidegrid): #+1
            amplphon[k, l] = np.mean(amplphonusam[k*factusam:(k+1)*factusam, l*factusam:(l+1)*factusam])
    indx = np.linspace(factusam / 2, numbsidegridusam - factusam / 2 + 1, numbsidegrid, dtype=int)

    xpostemp = xpostemp[indx]
    ypostemp = ypostemp[indx]

    xpos, ypos = np.meshgrid(xpostemp, ypostemp)

    amplphon = amplphon.flatten()
    ## normalize such that the sum is unity
    amplphon /= sum(amplphon)

    gridphon = np.vstack((xpos.flatten(), ypos.flatten()))

    return gridphon, amplphon

def retr_tranphon(gridphon, amplphon, galaxypars, convert=True):
    xposgalx, yposgalx, fluxgalx, arg1, arg2, arg3 = galaxypars
    if convert:
        radigalx, thetagalx, phigalx = from_moments(arg1, arg2, arg3)
        radigalx *= 1.33846609401 / np.sqrt(2) # only works for Sersic index 2, makes <x^2>=1
    else:
        radigalx, thetagalx, phigalx = arg1, arg2, arg3
    tranmatr = np.empty((xposgalx.size, 2, 2))
    tranmatr[:,0,0] = radigalx * np.cos(phigalx)
    tranmatr[:,0,1] = -radigalx * np.cos(thetagalx) * np.sin(phigalx)
    tranmatr[:,1,0] = radigalx * np.sin(phigalx)
    tranmatr[:,1,1] = radigalx * np.cos(thetagalx) * np.cos(phigalx)

    # transform the phonion grid
    gridphontran = np.matmul(tranmatr, gridphon)

    # positions of the phonions
    xposphon = xposgalx[:, None] + gridphontran[:, 0, :]
    yposphon = yposgalx[:, None] + gridphontran[:, 1, :]

    # spectrum of the phonions
    specphon = fluxgalx[:, None] * amplphon[None, :]

    return (xposphon.flatten()).astype(np.float32), (yposphon.flatten()).astype(np.float32), (specphon.flatten()).astype(np.float32)
