import numpy as np
from scipy.stats import scoreatpercentile as scpc
import h5py

import transitmodel as tmod
import emcee
import time as thetime

from emcee.utils import MPIPool
import sys
from astropy.io import fits 
import lightkurve
from tqdm import tqdm

filename = '../data/hlsp_tess-data-alerts_tess_phot_00307210830-s02_tess_v1_lc.fits'
DEFAULT_BITMASK = lightkurve.TessQualityFlags.DEFAULT_BITMASK


def get_lc():
    f = fits.open(filename)
    time_orig = f[1].data.TIME
    flux_orig = f[1].data.PDCSAP_FLUX
    quality = f[1].data.QUALITY

    time = time_orig[((quality & DEFAULT_BITMASK) == 0) &
                     (np.isfinite(time_orig)) & (np.isfinite(flux_orig))]
    flux = flux_orig[((quality & DEFAULT_BITMASK) == 0) &
                     (np.isfinite(time_orig)) & (np.isfinite(flux_orig))]
    
    mflux = (flux / np.median(flux)) - 1.0
    ferr = np.ones_like(mflux)  * np.std(mflux) * 0.3

    return time, mflux, ferr

def main(runmpi=False, nw=100, th=6, bi=10, fr=10):

    if runmpi:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool=None


    time, flux, ferr  = get_lc()

    toi = 175
    cadence = 120

    rho = 18
    rho_unc = 1
    nplanets = 3

    ld1 = 0.1642
    ld2 = 0.4259

    dil=0.0

    periods = [2.25321888449, 3.6906274382, 7.45131144274]
    impacts = [0.26, 0.21, 0.89]
    T0s = [1354.90455205, 1356.203624274, 1355.2866249]
    rprss = [0.02011, 0.038564, 0.0438550698]

    planet_guess = np.array([])
    for i in range(nplanets):
        planet_guess = np.r_[planet_guess,
                             T0s[i], periods[i], impacts[i], rprss[i],
                             0.0, 0.0
                             ]

    nwalkers = nw
    threads = th
    burnin = bi
    fullrun = fr
    thin = 1

    M = tmod.transitmc2(
        nplanets, cadence)

    M.get_ld(ld1, ld2)

    M.already_open(time,
        flux, ferr)

    M.get_rho([rho, rho_unc])
    M.get_zpt(0.0)

    M.get_sol(*planet_guess)

    outfile = 'koi{0}_np{1}.hdf5'.format(
            toi, nplanets)

    p0 = M.get_guess(nwalkers)
    l_var = np.shape(p0)[1]

    tom = tmod.logchi2
    args = [M.nplanets, M.rho_0, M.rho_0_unc,
            M.ld1, M.ld1_unc, M.ld2, M.ld2_unc,
            M.flux, M.err, 
            M.fixed_sol,
            M.time, M._itime, M._ntt,
            M._tobs, M._omc, M._datatype]

    N = len([indval for indval in range(fullrun)
            if indval%thin == 0])
    with h5py.File(outfile, u"w") as f:
        f.create_dataset("time", data=M.time)
        f.create_dataset("flux", data=M.flux)
        f.create_dataset("err", data=M.err)
        f.attrs["rho_0"] = M.rho_0
        f.attrs["rho_0_unc"] = M.rho_0_unc
        f.attrs["nplanets"] = M.nplanets
        f.attrs["ld1"] = M.ld1
        f.attrs["ld2"] = M.ld2
        g = f.create_group("mcmc")
        g.attrs["nwalkers"] = nwalkers
        g.attrs["burnin"] = burnin
        g.attrs["iterations"] = fullrun
        g.attrs["thin"] = thin
        g.create_dataset("fixed_sol", data= M.fixed_sol)
        g.create_dataset("fit_sol_0", data= M.fit_sol_0)


        c_ds = g.create_dataset("chain",
            (nwalkers, N, l_var),
            dtype=np.float64)
        lp_ds = g.create_dataset("lnprob",
            (nwalkers, N),
            dtype=np.float64)

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
    for i, (pos, lnprob, state) in enumerate(tqdm(sampler.sample(p2,
        iterations=fullrun, rstate0=state,
        storechain=False), total=fullrun)):

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
    sampler = main(runmpi=True,nw=110,th=14,bi=1,fr=5000)
