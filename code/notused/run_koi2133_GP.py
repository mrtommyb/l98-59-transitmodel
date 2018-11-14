from __future__ import division, print_function

#from clean_and_search import Clean
import numpy as np
from scipy.stats import scoreatpercentile as scpc
import h5py
from scipy.stats import nanmedian, nanstd
#import pyfits
#import kplr

import transitemcee_koi2133 as tmod
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


def main(runmpi=True,nw=100,th=6,bi=10,fr=10):

    if runmpi:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool=None

    ldfileloc = '/Users/tom/Projects/koi2133/code/'
    codedir = '/Users/tom/Projects/koi2133/code'

    koi = 2133
    cadence=1625.3
    teff = 4550.
    teff_unc = 75.
    feh = 0.11
    feh_unc = 0.07
    logg = 2.943
    logg_unc = 0.007
    rho = 0.0073
    rho_unc = 0.0001
    nplanets = 1

    dil=0.0

    period=6.24672336
    impact=0.2
    T0=136.383880
    rprs=0.02067
    alb=30.
    occ=40.
    ell=100.
    rvamp=79.0
    ecosw=0.048
    esinw=-0.045
    planet_guess = np.array([
        T0,period,impact,rprs,ecosw,esinw,
        rvamp,occ,ell,alb])


    rvtime, rvval, rverr = get_rv()
    time,flux,ferr = get_lc()

    rho_prior = True
    ldp_prior = False

    nwalkers = nw
    threads = th
    burnin = bi
    fullrun = fr
    thin = 1

    n_ldparams = 2

    toffset_lc = 0
    toffset_rv = 0

    zpt_0 = 1.E-10

    M = tmod.transitemcee_koi2133(
        nplanets,cadence,
        ldfileloc=ldfileloc,codedir=codedir)

    M.get_stellar(teff,
        logg,
        feh,
        n_ldparams,ldp_prior=ldp_prior)

    M.already_open(time,
        flux,ferr,
        rvtime,rvval,
        rverr,
        timeoffset=toffset_lc,rvtimeoffset=toffset_rv,
        normalize=False)

    rho_vals = np.array([rho,rho_unc])

    M.get_rho(rho_vals,rho_prior)
    M.get_zpt(zpt_0)

    noise_model = [2.1E-4, 3.2E-2, 2.E-4, 3.]


    if dil is not None:
        M.get_sol(*planet_guess,dil=dil,noise_model=noise_model)
    else:
        M.get_sol(*planet_guess,noise_model=noise_model)


    outfile = 'koi{0}_np{1}_prior{2}_dil{3}GP.hdf5'.format(
            koi,nplanets,rho_prior,dil)

    p0 = M.get_guess(nwalkers)

    #dirty hack!!
    qwe = np.r_[np.arange(0,5),np.arange(9,21)]
    p0 = p0[:,qwe]

    l_var = np.shape(p0)[1]

    N = len([indval for indval in xrange(fullrun)
            if indval%thin == 0])
    with h5py.File(outfile, u"w") as f:
        f.create_dataset("time", data=M.time)
        f.create_dataset("flux", data=M.flux)
        f.create_dataset("err", data=M.err)
        f.create_dataset("rvtime", data=M.rvtime)
        f.create_dataset("rvval", data=M.rvval)
        f.create_dataset("rverr", data=M.rverr)
        f.create_dataset("itime", data=M._itime)
        f.create_dataset("ntt", data = M._ntt)
        f.create_dataset("tobs", data = M._tobs)
        f.create_dataset("omc",data = M._omc)
        f.create_dataset("datatype",data = M._datatype)
        f.attrs["rho_0"] = M.rho_0
        f.attrs["rho_0_unc"] = M.rho_0_unc
        f.attrs["nplanets"] = M.nplanets
        f.attrs["ld1"] = M.ld1
        f.attrs["ld2"] = M.ld2
        f.attrs["koi"] = koi
        f.attrs["dil"] = dil
        g = f.create_group("mcmc")
        g.attrs["nwalkers"] = nwalkers
        g.attrs["burnin"] = burnin
        g.attrs["iterations"] = fullrun
        g.attrs["thin"] = thin
        g.attrs["rho_prior"] = rho_prior
        g.attrs["ldp_prior"] = ldp_prior
        g.attrs["onlytransits"] = M.onlytransits
        g.attrs["tregion"] = M.tregion
        g.attrs["ldfileloc"] = M.ldfileloc
        g.attrs["n_ldparams"] = M.n_ldparams
        g.create_dataset("fixed_sol", data= M.fixed_sol)
        g.create_dataset("fit_sol_0", data= M.fit_sol_0)


        c_ds = g.create_dataset("chain",
            (nwalkers, N, l_var),
            dtype=np.float64)
        lp_ds = g.create_dataset("lnprob",
            (nwalkers, N),
            dtype=np.float64)

        #I don't like the default LDP unc
        #I'm changing them
        M.ld1_unc = 0.8
        M.ld2_unc = 0.8

        args = [M.nplanets,M.rho_0,M.rho_0_unc,M.rho_prior,
            M.ld1,M.ld1_unc,M.ld2,M.ld2_unc,M.ldp_prior,
            M.flux,M.err,M.fixed_sol,M.time,M._itime,M._ntt,
            M._tobs,M._omc,M._datatype,
            M.rvtime,M.rvval,M.rverr,M._rvitime,
            M.n_ldparams,M.ldfileloc,
            M.onlytransits,M.tregion]


        tom = tmod.logchi2_rv_phaseGP_HP


        if runmpi:
            sampler = emcee.EnsembleSampler(nwalkers, l_var, tom,
                args=args,pool=pool)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, l_var, tom,
                args=args,threads=th)


        time1 = thetime.time()
        p2, prob, state = sampler.run_mcmc(p0, burnin,
            storechain=False)
        sampler.reset()
        with h5py.File(outfile, u"a") as f:
            g = f["mcmc"]
            g.create_dataset("burnin_pos", data=p2)
            g.create_dataset("burnin_prob", data=prob)

        time2 = thetime.time()
        print('burn-in took ' + str((time2 - time1)/60.) + ' min')
        time1 = thetime.time()
        for i, (pos, lnprob, state) in enumerate(sampler.sample(p2,
            iterations=fullrun, rstate0=state,
            storechain=False)):

            #do the thinning in the loop here
            if i % thin == 0:
                ind = i / thin
                with h5py.File(outfile, u"a") as f:
                    g = f["mcmc"]
                    c_ds = g["chain"]
                    lp_ds = g["lnprob"]
                    c_ds[:, ind, :] = pos
                    lp_ds[:, ind] = lnprob

        time2 = thetime.time()
        print('MCMC run took ' + str((time2 - time1)/60.) + ' min')
        print('')
        print("Mean acceptance: "
            + str(np.mean(sampler.acceptance_fraction)))
        print('')

        if runmpi:
            pool.close()
        else:
            sampler.pool.close()

        return sampler

if __name__ == '__main__':
    sampler = main(runmpi=True,nw=70,th=6,bi=1,fr=10)


