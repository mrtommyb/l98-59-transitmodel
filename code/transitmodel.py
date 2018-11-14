import emcee
import numpy as np
import tmodtom as tmod
import time as thetime
from scipy.stats import truncnorm
from emcee.utils import MPIPool
import tmodtom as tmod
from copy import deepcopy


class transitmc2(object):
    def __init__(self,nplanets,cadence=120):
        self.nplanets = nplanets
        nmax = 1500000 #from the fortran
        self._ntt = np.zeros(nplanets)
        self._tobs = np.empty([self.nplanets,nmax])
        self._omc = np.empty([self.nplanets,nmax])
        self.cadence = cadence / 86400.

    def get_ld(self, ld1, ld2):
        self.ld1 = ld1
        self.ld2 = ld2
        self.ld3 = 0.0
        self.ld4 = 0.0
        self.ld1_unc = 0.1
        self.ld2_unc = 0.1

    def already_open(self,t1, f1, e1):
        time = t1
        self.time = time
        self.flux = f1
        self.err = e1
        self.npt = len(time)
        self._itime = np.zeros(self.npt) + self.cadence
        self._datatype = np.zeros(self.npt)

    def get_rho(self, rho_vals):
        """
        inputs
        rho_vals : array_like
            Two parameter array with value
            rho, rho_unc
        prior : bool, optional
            should this rho be used as a prior?
        """
        self.rho_0 = rho_vals[0]
        self.rho_0_unc = rho_vals[1]



    def get_zpt(self,zpt_0=1.E-10):
        self.zpt_0 = zpt_0
        if self.zpt_0 == 0.0:
            self.zpt_0 = 1.E-10
        self.zpt_0_unc = 1.E-6

    def get_sol(self, *args):
        dil = 0.0
        veloffset = 0.0
        rvamp = 0.0
        occ = 0.0
        ell = 0.0
        alb = 0.0
        fit_sol = np.array([np.log(self.rho_0), self.zpt_0,
                self.ld1, self.ld2])

        for i in range(self.nplanets):
            T0_0 = args[i*6]
            per_0 = args[i*6 +1]
            b_0 = args[i*6 +2]
            rprs_0 = np.log(args[i*6 +3])
            ecosw_0 = args[i*6 +4]
            esinw_0 = args[i*6 +5]
            new_params = np.array([T0_0,per_0,
                b_0,rprs_0,ecosw_0,esinw_0])
            fit_sol = np.r_[fit_sol,new_params]

        self.fit_sol = fit_sol
        self.fit_sol_0 = deepcopy(self.fit_sol)
        self.fixed_sol = np.array([
            dil,veloffset,rvamp,
            occ,ell,alb])

    def get_guess(self,nwalkers):
        """
        pick sensible starting ranges for the guess parameters
        T0, period, impact paramter, rp/rs, ecosw and esinw
        """
        rho_unc = 0.1
        zpt_unc = 1.E-9
        ld1_unc = 0.01
        ld2_unc = 0.01
        T0_unc = 0.00002
        per_unc = 0.00005
        b_unc = 0.001
        rprs_unc = 0.0001
        ecosw_unc = 0.001
        esinw_unc = 0.001

        p0 = np.zeros([nwalkers,4+self.nplanets*6+1])


        logrho = self.fit_sol[0]
        zpt = self.fit_sol[1]
        ld1 = self.fit_sol[2]
        ld2 = self.fit_sol[3]


        start,stop = (-10 - logrho) / rho_unc, (5 - logrho) / rho_unc
        p0[...,0] = truncnorm.rvs(start,stop
                ,loc=logrho,scale=rho_unc,size=nwalkers)

        p0[...,1] = np.random.normal(loc=zpt,scale=zpt,size=nwalkers)

        start,stop = (0.0 - ld1) / ld1_unc, (1.0 - ld1) / ld1_unc
        p0[...,2] = truncnorm.rvs(start,stop
                ,loc=ld1,scale=ld1_unc,size=nwalkers)

        start,stop = (0.0 - ld2) / ld2_unc, (1.0 - ld2) / ld2_unc
        p0[...,3] = truncnorm.rvs(start,stop
                ,loc=ld2,scale=ld2_unc,size=nwalkers)


        for i in range(self.nplanets):
            (T0, per, b, logrprs, ecosw,
                esinw) = self.fit_sol[i*6+4:i*6 + 10]
            ecosw = 0.0
            esinw = 0.0
            p0[...,i*6+4] = np.random.normal(
                T0, T0_unc, size=nwalkers)
            p0[...,i*6+5] = np.random.normal(
                per,per_unc, size=nwalkers)
            start, stop = (0.0 - b) / b_unc, (0.95 - b) / b_unc
            p0[...,i*6+6] = truncnorm.rvs(
                start,stop
                ,loc=b, scale=b_unc, size=nwalkers)
            start, stop = (-7 - logrprs) / rprs_unc, (-0.7 - logrprs) / rprs_unc
            p0[...,i*6+7] = truncnorm.rvs(
                start, stop,
                loc=logrprs, scale=rprs_unc, size=nwalkers)
            start,stop = (0.0 - ecosw) / ecosw_unc, (0.5 - ecosw) / ecosw_unc
            p0[...,i*6+8] = truncnorm.rvs(
                start,stop, loc=ecosw, scale=ecosw_unc, size=nwalkers)
            start,stop = (0.0 - esinw) / esinw_unc, (0.5 - esinw) / esinw_unc
            p0[...,i*6+9] = truncnorm.rvs(
                start,stop, loc=esinw, scale=esinw_unc, size=nwalkers)

        #this is the jitter term
        #make it like self.err
        p0[...,-1] = np.random.normal(
                -7.22, 0.1, size=nwalkers)
        return p0

def get_ar(rho,period):
    """ gets a/R* from period and mean stellar density"""
    G = 6.67E-11
    rho_SI = rho * 1000.
    tpi = 3. * np.pi
    period_s = period * 86400.
    part1 = period_s**2 * G * rho_SI
    ar = (part1 / tpi)**(1./3.)
    return ar

def logchi2(fitsol, nplanets, 
    rho_0, rho_0_unc,
    ld1_0, ld1_0_unc, ld2_0, ld2_0_unc,
    flux, err, fixed_sol, time, itime, ntt, tobs, omc, datatype):
    minf = -np.inf
    logrho = fitsol[0]
    rho = np.exp(logrho)
    if rho > 40.:
        return minf
    zpt = fitsol[1]
    ld1 = fitsol[2]
    ld2 = fitsol[3]
    # some lind darkening constraints
    # from Burke et al. 2008 (XO-2b)
    if ld1 < 0.0:
        return minf
    if ld1 + ld2 > 1.0:
        return minf
    if (ld1 + 2.) * (ld2 < 0.0):
        return minf
    if ld2 < -0.8:
        return minf

    ld3, ld4 = 0.0, 0.0
    lds = np.array([ld1,ld2,ld3,ld4])

    logrprs = fitsol[np.arange(nplanets)*6 + 7]
    rprs = np.exp(logrprs)
    if np.any(rprs < 0.) or np.any(rprs > 0.5):
            return minf

    ecosw = fitsol[np.arange(nplanets)*6 + 8]
    if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
        return minf

    esinw = fitsol[np.arange(nplanets)*6 + 9]
    if np.any(esinw < -1.0) or np.any(esinw > 1.0):
        return minf

    #avoid parabolic orbits
    ecc = np.sqrt(esinw**2 + ecosw**2)
    if np.any(ecc > 1.0):
        return minf
    ecc[ecc == 0.0] = 1.E-10

    #avoid orbits where the planet enters the star
    per = fitsol[np.arange(nplanets)*6 + 5]
    ar = get_ar(rho, per)
    if np.any(ecc > (1.-(1. / ar))):
        return minf

    # lets avoid orbit corssings
    if nplanets > 1:
        for i in range(1,nplanets):
            if (ar[i-1]*(1+ecc[i-1])) > (ar[i]*(1-ecc[i])):
                return minf

    b = fitsol[np.arange(nplanets)*6 + 6]
    if np.any(b < 0.) or np.any(b > 1.0 + rprs):
        return minf

    logjitter = fitsol[-1]
    jitter = np.exp(logjitter)
    if jitter > 0.07:
        return minf
    err_jit2 = err**2 + jitter**2
    err_jit = np.sqrt(err_jit2)
    npt_lc = len(err_jit)

    fitsol[np.arange(nplanets)*6 + 7] = np.exp(fitsol[np.arange(nplanets)*6 + 7])
    fitsol[-1] = np.exp(fitsol[-1])
    fitsol_model_calc = np.r_[rho, zpt, fitsol[4:]]
    fixed_sol_model_calc = np.r_[lds,fixed_sol]

    model_lc = calc_model(fitsol_model_calc,nplanets,fixed_sol_model_calc,
        time,itime,ntt,tobs,omc,datatype)

    loglc = (
        - (npt_lc/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(err_jit2))
        - 0.5 * np.sum((model_lc - flux)**2 / err_jit2)
        )

    logrho = (
        - 0.5 * np.log(2.*np.pi)
        - 0.5 * np.log(rho_0_unc**2)
        - 0.5 * (rho_0 - rho)**2 / rho_0_unc**2
        )

    logld1 = (
        - 0.5 * np.log(2.*np.pi)
        - 0.5 * np.log(ld1_0_unc**2)
        - 0.5 * (ld1_0 - ld1)**2 / ld1_0_unc**2
        )

    logld2 = (
        - 0.5 * np.log(2.*np.pi)
        - 0.5 * np.log(ld2_0_unc**2)
        - 0.5 * (ld2_0 - ld2)**2 / ld2_0_unc**2
        )

    logldp = logld1 + logld2

    logecc = - np.sum(np.log(ecc))

    logLtot = loglc + logrho + logldp + logecc

    return logLtot

def calc_model(fitsol, nplanets, fixed_sol, time, itime, ntt, tobs, omc, datatype):
    sol = np.zeros([8 + 10*nplanets])
    rho = fitsol[0]
    zpt = fitsol[1]
    ld1,ld2,ld3,ld4 = fixed_sol[0:4]
    dil = fixed_sol[4]
    veloffset = fixed_sol[5]

    fixed_stuff = fixed_sol[6:10]

    sol[0:8] = np.array([rho,ld1,ld2,ld3,ld4,
        dil,veloffset,zpt])
    for i in range(nplanets):
        sol[8+(i*10):8+(i*10)+10] = np.r_[fitsol[2+i*6:8+i*6],fixed_stuff]

    tmodout = tmod.transitmodel(nplanets,sol,time,itime,
        ntt,tobs,omc,datatype)

    return tmodout - 1.