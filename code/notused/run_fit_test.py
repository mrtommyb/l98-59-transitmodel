from __future__ import division, print_function

import numpy as np
from scipy.stats import scoreatpercentile as scpc
import h5py
from scipy.stats import nanmedian, nanstd
#import pyfits
#import kplr

#import transitemcee_koi2133 as tmod
import emcee
import time as thetime
#import os
from emcee.utils import MPIPool
import sys

def get_rv():
    rvdat = np.array([
        [2456041.86670891,  -9286.04,    24.31],
        [2456099.71041653,  -9378.65,    17.52],
        [2456100.70865078,  -9424.57,    19.74],
        [2456101.70698350,  -9445.55,    16.88],
        [2456104.68968502,  -9332.69,    18.45],
        [2456108.69264679,  -9425.55,    25.04],
        [2456133.86443801,  -9409.06,    27.11],
        [2456191.70858740,  -9303.60,    18.44],
        [2456215.63282948,  -9399.96,    14.93],
    ])
    rvtime = rvdat[...,0] - 2454833.
    rvval = rvdat[...,1] - np.median(rvdat[...,1])
    rverr = rvdat[...,2]
    return rvtime,rvval,rverr

def get_lc():
    time,flux,ferr = np.genfromtxt(
        '../data/lc.dat',unpack=True)
    return time,flux,ferr

if __name__ == '__main__':
    import ktransit
    from ktransit import FitTransit
    import matplotlib.pyplot as plt
    import numpy

    fitT = FitTransit()
    fitT.add_guess_star(
            rho=43.47,
            ld1=0.43,
            ld2=0.27,
            ld3=0.0,
            ld4=0.0,
            dil=0.0,
            zpt=-56.8E-6,
            veloffset=15.6 # new keyword, the radial velocity zero-point offset in m/s
            )
    fitT.add_guess_planet(
            T0=136.3837,
            period=6.246696,
            impact=0.865,
            rprs= 0.019429,
            ecosw=0.0134,
            esinw=-0.012,
            occ=45.,
            rvamp=70.2,
            ell=50.8,
            alb=27) # radial velocity semi-amplitude in m/s

    time,flux,ferr = get_lc()
    rvtime,rvval,rverr = get_rv()
    fitT.add_data(
            time=time,
            flux=flux,ferr=ferr)

    fitT.add_rv(rvtime=rvtime,                             # radial velocity observation timestamps
            rvval=rvval,rverr=rverr)    # integration time of each timestamp

    vary_star = ['rho', 'zpt']      # free stellar parameters
    vary_planet = (['period',       # free planetary parameters
        'T0', 'impact',
        'rprs'])                # free planet parameters are the same for every planet you model

    fitT.free_parameters(vary_star, vary_planet)
    fitT.do_fit()



    #plt.plot(fitT.time,fitT.tmod)
    #plt.plot(M.rvtime,rvmodel)






