import sys
import numpy as np
#import matplotlib.pyplot as plt
import emcee
import tmodtom as tmod
import time as thetime
from scipy.stats import truncnorm
from claretquadpy import claretquad
from claret4ppy import claretlimb4p
from copy import deepcopy
from numpy import random
#from bilin_interp import ld_quad



class transitemcee(object):
    def __init__(self,nplanets,cadence=1625.3,
        ldfileloc='/Users/tom/svn_code/tom_code/',
        codedir='/Users/tom/svn_code/tom_code/'):
        sys.path.append(codedir)
        self.nplanets = nplanets
        nmax = 1500000 #from the fortran
        self._ntt = np.zeros(nplanets)
        self._tobs = np.empty([self.nplanets,nmax])
        self._omc = np.empty([self.nplanets,nmax])
        self.cadence = cadence / 86400.
        self.allow_ecc_orbit = False
        self.ldfileloc = ldfileloc
        self.onlytransits = False
        self.tregion = 500


    def get_stellar(self,teff,logg,FeH,n_ldparams=4):
        """
        read in stellar parameters
        inputs
        teff : float
            The effective temperature of the star
        logg : float
            the surface gravity of the star in log cgs
        FeH : float
            the metalicity of the star in log solar

        optional
        n_ldparams : int
        """
        self.Teff = teff
        self.logg = logg
        self.FeH = FeH
        if n_ldparams == 2:
            #if teff < 3500 and logg >= 3.5:
            if False:
                #this block should never run
                ldfile = self.ldfileloc + 'claret-quad-phoenix.txt'
                self.ld1,self.ld2 = ld_quad(ldfile,
                    self.Teff,self.logg)
                self.ld3 = 0.0
                self.ld4 = 0.0

            #elif logg < 3.5 or teff >= 3500:
            if True:
                ldfile = self.ldfileloc + 'claret-limb-quad.txt'
                self.ld1,self.ld2 = claretquad(ldfile,
                    self.Teff,self.logg,self.FeH)
                self.ld3 = 0.0
                self.ld4 = 0.0

        elif n_ldparams == 4:
            ldfile = self.ldfileloc + 'claret-limb.txt'
            self.ld1,self.ld2,self.ld3,self.ld4 = claretlimb4p(ldfile,
                self.Teff,self.logg,self.FeH)

    def open_lightcurve(self,filename,timeoffset=0.0,
        normalize=False):
        t = np.genfromtxt(filename).T
        time = t[0] - timeoffset
        if normalize:
            flux = t[1] / np.median(t[1])
            err = t[2] / np.median(t[1])
        else:
            flux = t[1]
            err = t[2]
        self.time = time
        self.flux = flux
        self.err = err
        self.npt = len(time)
        self._itime = np.zeros(self.npt) + self.cadence
        self._datatype = np.zeros(self.npt)

    def already_open(self,t1,f1,e1,timeoffset=0.0,normalize=False):
        time = t1 - timeoffset
        if normalize:
            flux = f1 / np.median(f1)
            err = e1 / np.median(f1)
        else:
            flux = f1
            err = e1
        self.time = time
        self.flux = flux
        self.err = err
        self.npt = len(time)
        self._itime = np.zeros(self.npt) + self.cadence
        self._datatype = np.zeros(self.npt)

    def get_rho(self,rho_vals,prior=False,rho_start=0.0,
        rho_stop = 30.):
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
        self.rho_0_start = rho_start
        self.rho_0_stop = rho_stop

        if prior:
            self.rho_prior = True
        else:
            self.rho_prior = False

    def get_zpt(self,zpt_0):
        self.zpt_0 = zpt_0
        if self.zpt_0 == 0.0:
            self.zpt_0 = 1.E-10


    def get_sol(self,*args,**kwargs):
        """
        reads the guess transit fit solution

        There are 6 args for every planet
        T0, period, impact paramter, rp/rs, ecosw and esinw

        optional keywords, these are kept fixed (for now)
        dil : float, optional
            dilution
        veloffset : float, optional
            velocity zeropoint
        rvamp : float, optional
            radial velocity amplitude from doppler beaming
        occ : float, optional
            occultation depth
        ell : float, optional
            amplitude of ellipsoidal variations
        alb : float, optional
            geometric albedo of the planet
        """

        assert len(args) == self.nplanets * 6

        if 'dil' in kwargs.keys():
            dil = kwargs['dil']
            print ' running with dil = %s' %(dil)
        else:
            dil = 0.0
        if 'veloffset' in kwargs.keys():
            veloffset = kwargs['veloffset']
        else:
            veloffset = 0.0
        if 'rvamp' in kwargs.keys():
            rvamp = kwargs['rvamp']
        else:
            rvamp = 0.0
        if 'occ' in kwargs.keys():
            occ = kwargs['occ']
        else:
            occ = 0.0
        if 'ell' in kwargs.keys():
            ell = kwargs['ell']
        else:
            ell = 0.0
        if 'alb' in kwargs.keys():
            alb = kwargs['alb']
        else:
            alb = 0.0

        try:
            if self.zpt_0 == 0.:
                self.zpt_0 = 1.E-10
        except AttributeError:
            self.zpt_0 = 1.E-10
            self.zpt_0_unc = 1.E-6


        fit_sol = np.array([self.rho_0,self.zpt_0])
        for i in xrange(self.nplanets):
            T0_0 = args[i*6]
            per_0 = args[i*6 +1]
            b_0 = args[i*6 +2]
            rprs_0 = args[i*6 +3]
            ecosw_0 = args[i*6 +4]
            esinw_0 = args[i*6 +5]

            new_params = np.array([T0_0,per_0,
                b_0,rprs_0,ecosw_0,esinw_0])
            fit_sol = np.r_[fit_sol,new_params]

        self.fit_sol = fit_sol
        self.fit_sol_0 = deepcopy(self.fit_sol)

        self.fixed_sol = np.array([self.ld1,self.ld2,
            self.ld3,self.ld4,
            dil,veloffset,rvamp,
            occ,ell,alb])

    def cut_non_transit(self,ntdur=10):
        #make a mask for each planet candidate
        self.onlytransits = True
        tregion = np.zeros(self.nplanets)

        maskdat = np.zeros([self.npt,self.nplanets],dtype=bool)

        for i in xrange(self.nplanets):
            T0 = self.fit_sol[i*6 + 2]
            per = self.fit_sol[i*6 + 3]
            rho = self.fit_sol[0]

            ars = self.get_ar(rho,per)
            tdur_dys = (1./ars) * per * (1./np.pi)

            #this is buggy because T0 is not nessessarily time of first transit
            #but time of a transit. So fudge.
            #subtract make T0 the first transit
            time0 = np.copy(T0)
            while True:
                if time0 - per < self.time[0]:
                    break
                else:
                    time0 = time0 - per

            ntransits = int((self.time[-1] - self.time[0]) / per) + 1
            t_times = np.arange(ntransits)*per + T0

            #make sure the first and last transit are not excluded even if
            #partially in the data
            t_times = np.r_[t_times,t_times[0] - per,t_times[-1] + per]

            for j in t_times:
                maskdat[:,i] = np.logical_or(maskdat[:,i],
                    np.logical_and(
                    self.time < j +tdur_dys*ntdur,
                    self.time > j - tdur_dys*ntdur) )
            tregion[i] = ntdur*tdur_dys
        #create a final mask that is the OR of the
        #individual masks
        finmask = np.zeros(self.npt)
        for i in xrange(self.nplanets):
            finmask = np.logical_or(finmask,maskdat[:,i])

        self.time = self.time[finmask]
        self.flux = self.flux[finmask]
        self.err = self.err[finmask]
        self._itime = self._itime[finmask]
        self._datatype = self._datatype[finmask]
        self.tregion = tregion

    def get_ar(self,rho,period):
        """ gets a/R* from period and mean stellar density"""
        G = 6.67E-11
        rho_SI = rho * 1000.
        tpi = 3. * np.pi
        period_s = period * 86400.
        part1 = period_s**2 * G * rho_SI
        ar = (part1 / tpi)**(1./3.)
        return ar

    # def calc_model(self,fitsol):
    #     sol = np.zeros([8 + 10*self.nplanets])
    #     rho = fitsol[0]
    #     zpt = fitsol[1]
    #     ld1,ld2,ld3,ld4 = self.fixed_sol[0:4]
    #     dil = self.fixed_sol[4]
    #     veloffset = self.fixed_sol[5]

    #     fixed_stuff = self.fixed_sol[6:10]

    #     sol[0:8] = np.array([rho,ld1,ld2,ld3,ld4,
    #         dil,veloffset,zpt])
    #     for i in xrange(self.nplanets):
    #         sol[8+(i*10):8+(i*10)+10] = np.r_[fitsol[2+i*6:8+i*6],fixed_stuff]

    #     tmodout = tmod.transitmodel(self.nplanets,sol,self.time,self._itime,
    #         self._ntt,self._tobs,self._omc,self._datatype)

    #     return tmodout - 1.

    # def logchi2(self,fitsol):
    #     rho = fitsol[0]
    #     if rho < 0.001 or rho > 30.:
    #         return -np.inf
    #     rprs = fitsol[np.arange(self.nplanets)*6 + 5]
    #     if np.any(rprs < 0.) or np.any(rprs > 0.5):
    #         return -np.inf
    #     ecosw = fitsol[np.arange(self.nplanets)*6 + 6]
    #     if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
    #         return -np.inf
    #     esinw = fitsol[np.arange(self.nplanets)*6 + 7]
    #     if np.any(esinw < -1.0) or np.any(esinw > 1.0):
    #         return -np.inf
    #     b = fitsol[np.arange(self.nplanets)*6 + 4]
    #     if np.any(b < 0.) or np.any(b > 1.0 + rprs):
    #         return -np.inf


    #     model_lc = self.calc_model(fitsol)

    #     if self.rho_prior:
    #         chi2prior = (self.rho_0 - rho)**2 / self.rho_0_unc**2
    #     else:
    #         chi2prior = 0.0

    #     chi2val = np.sum((model_lc - self.flux)**2 / self.err**2)
    #     chi2tot = chi2val + chi2prior
    #     logp = -chi2tot / 2.
    #     return logp

    # def do_emcee(self,nwalkers,threads=16,burnin=100,fullrun=1000):
    #     l_var = 8
    #     p0 = self.get_guess(nwalkers)

    #     sampler = emcee.EnsembleSampler(nwalkers, l_var, self.logchi2,
    #     threads=threads)

    #     time1 = thetime.time()
    #     pos, prob, state = sampler.run_mcmc(p0, burnin)
    #     sampler.reset()
    #     time2 = thetime.time()
    #     print 'burn-in took ' + str((time2 - time1)/60.) + ' min'

    #     time1 = thetime.time()
    #     sampler.run_mcmc(pos, fullrun)
    #     time2 = thetime.time()
    #     print 'MCMC run took ' + str((time2 - time1)/60.) + ' min'
    #     print
    #     print("Mean acceptance: "
    #         + str(np.mean(sampler.acceptance_fraction)))
    #     print

    #     try:
    #         print("Autocorrelation times sampled:", fullrun / sampler.acor)
    #     except RuntimeError:
    #         print("No Autocorrelation")

    #     return sampler, (time2 - time1)/60.

    def get_guess(self,nwalkers):
        """
        pick sensible starting ranges for the guess parameters
        T0, period, impact paramter, rp/rs, ecosw and esinw
        """
        rho_unc = 0.001
        zpt_unc = 1.E-8
        T0_unc = 0.0002
        per_unc = 0.00005
        b_unc = 0.001
        rprs_unc = 0.0001
        ecosw_unc = 0.001
        esinw_unc = 0.001
        p0 = np.zeros([nwalkers,2+self.nplanets*6])

        rho = self.fit_sol[0]
        zpt = self.fit_sol[1]

        start,stop = (0.0001 - rho) / rho_unc, (30.0 - rho) / rho_unc
        p0[...,0] = truncnorm.rvs(start,stop
                ,loc=rho,scale=rho_unc,size=nwalkers)

        p0[...,1] = np.random.normal(loc=zpt,scale=zpt,size=nwalkers)

        for i in xrange(self.nplanets):
            T0,per,b,rprs,ecosw,esinw = self.fit_sol[i*6+2:i*6 + 8]
            b = 0.0
            ecosw = 0.0
            esinw = 0.0
            p0[...,i*6+2] = np.random.normal(T0,T0_unc,size=nwalkers)
            p0[...,i*6+3] = np.random.normal(per,per_unc,size=nwalkers)
            start,stop = (0.0 - b) / b_unc, (0.5 - b) / b_unc
            p0[...,i*6+4] = truncnorm.rvs(start,stop
                ,loc=b,scale=b_unc,size=nwalkers)
            start,stop = (0.0 - rprs) / rprs_unc, (0.5 - rprs) / rprs_unc
            p0[...,i*6+5] = truncnorm.rvs(start,stop
                ,loc=rprs,scale=rprs_unc,size=nwalkers)
            start,stop = (0.0 - ecosw) / ecosw_unc, (0.5 - ecosw) / ecosw_unc
            p0[...,i*6+6] = truncnorm.rvs(start,stop
                ,loc=ecosw,scale=ecosw_unc,size=nwalkers)
            start,stop = (0.0 - esinw) / esinw_unc, (0.5 - esinw) / esinw_unc
            p0[...,i*6+7] = truncnorm.rvs(start,stop
                ,loc=esinw,scale=esinw_unc,size=nwalkers)

        return p0


class transitemcee_paramprior(transitemcee):

    def __init__(self,nplanets,cadence=1626.3,
        ldfileloc='/Users/tom/svn_code/tom_code/'):

        transitemcee.__init__(self,nplanets,cadence,ldfileloc)

    def get_stellar(self,teff,teff_unc,logg,logg_unc,FeH,FeH_unc,
        n_ldparams=2):
        """
        read in stellar parameters
        inputs
        teff : float
            The effective temperature of the star
        logg : float
            the surface gravity of the star in log cgs
        FeH : float
            the metalicity of the star in log solar

        optional
        n_ldparams : int
        """
        self.Teff = teff
        self.Teff_unc = teff_unc
        self.logg = logg
        self.logg_unc = logg_unc
        self.FeH = FeH
        self.FeH_unc = FeH_unc

        self.n_ldparams = n_ldparams



    def get_sol(self,*args,**kwargs):
        """
        reads the guess transit fit solution

        There are 6 args for every planet
        T0, period, impact paramter, rp/rs, ecosw and esinw

        optional keywords, these are kept fixed (for now)
        dil : float, optional
            dilution
        veloffset : float, optional
            velocity zeropoint
        rvamp : float, optional
            radial velocity amplitude from doppler beaming
        occ : float, optional
            occultation depth
        ell : float, optional
            amplitude of ellipsoidal variations
        alb : float, optional
            geometric albedo of the planet
        """

        assert len(args) == self.nplanets * 6

        if 'dil' in kwargs.keys():
            dil = kwargs['dil']
            print ' running with dil = %s' %(dil)
        else:
            dil = 0.0
        if 'veloffset' in kwargs.keys():
            veloffset = kwargs['veloffset']
        else:
            veloffset = 0.0
        if 'rvamp' in kwargs.keys():
            rvamp = kwargs['rvamp']
        else:
            rvamp = 0.0
        if 'occ' in kwargs.keys():
            occ = kwargs['occ']
        else:
            occ = 0.0
        if 'ell' in kwargs.keys():
            ell = kwargs['ell']
        else:
            ell = 0.0
        if 'alb' in kwargs.keys():
            alb = kwargs['alb']
        else:
            alb = 0.0

        try:
            if self.zpt_0 == 0.:
                self.zpt_0 = 1.E-10
        except AttributeError:
            self.zpt_0 = 1.E-10
            self.zpt_0_unc = 1.E-6


        fit_sol = np.array([self.rho_0,self.zpt_0,self.Teff,self.logg,self.FeH])
        for i in xrange(self.nplanets):
            T0_0 = args[i*6]
            per_0 = args[i*6 +1]
            b_0 = args[i*6 +2]
            rprs_0 = args[i*6 +3]
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
        rho_unc = 0.001
        zpt_unc = 1.E-8
        teff_unc = 10
        logg_unc = 0.01
        feh_unc = 0.01
        T0_unc = 0.0002
        per_unc = 0.00005
        b_unc = 0.001
        rprs_unc = 0.0001
        ecosw_unc = 0.001
        esinw_unc = 0.001
        p0 = np.zeros([nwalkers,5+self.nplanets*6])

        rho = self.fit_sol[0]
        zpt = self.fit_sol[1]
        teff = self.fit_sol[2]
        logg = self.fit_sol[3]
        feh = self.fit_sol[4]

        start,stop = (0.0001 - rho) / rho_unc, (30.0 - rho) / rho_unc
        p0[...,0] = truncnorm.rvs(start,stop
                ,loc=rho,scale=rho_unc,size=nwalkers)

        p0[...,1] = np.random.normal(loc=zpt,scale=zpt,size=nwalkers)

        start,stop = (3500. - teff) / teff_unc, (50000. - teff) / teff_unc
        p0[...,2] = truncnorm.rvs(start,stop
                ,loc=teff,scale=teff_unc,size=nwalkers)

        start,stop = (0.0 - logg) / logg_unc, (5. - logg) / logg_unc
        p0[...,3] = truncnorm.rvs(start,stop
                ,loc=logg,scale=logg_unc,size=nwalkers)

        start,stop = (-5.0 - feh) / feh_unc, (1.0 - feh) / feh_unc
        p0[...,4] = truncnorm.rvs(start,stop
                ,loc=feh,scale=feh_unc,size=nwalkers)

        for i in xrange(self.nplanets):
            T0,per,b,rprs,ecosw,esinw = self.fit_sol[i*6+5:i*6 + 11]
            b = 0.0
            ecosw = 0.0
            esinw = 0.0
            p0[...,i*6+5] = np.random.normal(T0,T0_unc,size=nwalkers)
            p0[...,i*6+6] = np.random.normal(per,per_unc,size=nwalkers)
            start,stop = (0.0 - b) / b_unc, (0.5 - b) / b_unc
            p0[...,i*6+7] = truncnorm.rvs(start,stop
                ,loc=b,scale=b_unc,size=nwalkers)
            start,stop = (0.0 - rprs) / rprs_unc, (0.5 - rprs) / rprs_unc
            p0[...,i*6+8] = truncnorm.rvs(start,stop
                ,loc=rprs,scale=rprs_unc,size=nwalkers)
            start,stop = (0.0 - ecosw) / ecosw_unc, (0.5 - ecosw) / ecosw_unc
            p0[...,i*6+9] = truncnorm.rvs(start,stop
                ,loc=ecosw,scale=ecosw_unc,size=nwalkers)
            start,stop = (0.0 - esinw) / esinw_unc, (0.5 - esinw) / esinw_unc
            p0[...,i*6+10] = truncnorm.rvs(start,stop
                ,loc=esinw,scale=esinw_unc,size=nwalkers)

        return p0

    def cut_non_transit(self,ntdur=10):
        #make a mask for each planet candidate
        self.onlytransits = True
        tregion = np.zeros(self.nplanets)

        maskdat = np.zeros([self.npt,self.nplanets],dtype=bool)

        for i in xrange(self.nplanets):
            T0 = self.fit_sol[i*6 + 5]
            per = self.fit_sol[i*6 + 6]
            rho = self.fit_sol[0]

            ars = self.get_ar(rho,per)
            tdur_dys = (1./ars) * per * (1./np.pi)

            #this is buggy because T0 is not nessessarily time of first transit
            #but time of a transit. So fudge.
            #subtract make T0 the first transit
            time0 = np.copy(T0)
            while True:
                if time0 - per < self.time[0]:
                    break
                else:
                    time0 = time0 - per

            ntransits = int((self.time[-1] - self.time[0]) / per) + 1
            t_times = np.arange(ntransits)*per + T0

            #make sure the first and last transit are not excluded even if
            #partially in the data
            t_times = np.r_[t_times,t_times[0] - per,t_times[-1] + per]

            for j in t_times:
                maskdat[:,i] = np.logical_or(maskdat[:,i],
                    np.logical_and(
                    self.time < j +tdur_dys*ntdur,
                    self.time > j - tdur_dys*ntdur) )
            tregion[i] = ntdur*tdur_dys
        #create a final mask that is the OR of the
        #individual masks
        finmask = np.zeros(self.npt)
        for i in xrange(self.nplanets):
            finmask = np.logical_or(finmask,maskdat[:,i])

        self.time = self.time[finmask]
        self.flux = self.flux[finmask]
        self.err = self.err[finmask]
        self._itime = self._itime[finmask]
        self._datatype = self._datatype[finmask]
        self.tregion = tregion


class transitemcee_paramprior_occ(transitemcee_paramprior):
    pass

class transitemcee_fitldp(transitemcee):

    def __init__(self,nplanets,cadence=1626.3,
        ldfileloc='/Users/tom/svn_code/tom_code/',
        codedir='/Users/tom/svn_code/tom_code/'):

        transitemcee.__init__(self,nplanets,cadence,ldfileloc,codedir)

    def get_stellar(self,teff,logg,FeH,
        n_ldparams=2,ldp_prior=True):
        """
        read in stellar parameters
        inputs
        teff : float
            The effective temperature of the star
        logg : float
            the surface gravity of the star in log cgs
        FeH : float
            the metalicity of the star in log solar

        optional
        n_ldparams : int
        """
        self.Teff = teff

        self.logg = logg

        self.FeH = FeH


        self.ld1_unc = 0.1
        self.ld2_unc = 0.1
        self.ld3_unc = 0.1
        self.ld4_unc = 0.1
        if teff < 3500:
            teff = 3500
            self.ld1_unc = 0.2
            self.ld2_unc = 0.2
        if logg < 0.0:
            logg = 0.0
            self.ld1_unc = 0.05
            self.ld2_unc = 0.05
        if logg > 5.0:
            logg = 5.0
            self.ld1_unc = 0.05
            self.ld2_unc = 0.05
        if FeH < -5.0:
            FeH = -5.0
            self.ld1_unc = 0.05
            self.ld2_unc = 0.05
        if FeH > 1.0:
            FeH = 1.0
            self.ld1_unc = 0.05
            self.ld2_unc = 0.05

        if n_ldparams == 2:
            ldfile = self.ldfileloc + 'claret-limb-quad.txt'
            self.ld1,self.ld2 = claretquad(ldfile,
                    teff,logg,FeH)
            self.ld3 = 0.0
            self.ld4 = 0.0
            if teff < 3500:
                self.ld1,self.ld2 = claretquad(ldfile,
                    3500.,logg,FeH)
        elif n_ldparams == 4:
            ldfile = self.ldfileloc + 'claret-limb.txt'
            self.ld1,self.ld2,self.ld3,self.ld4 = claretlimb4p(
                ldfile,
                self.Teff,self.logg,self.FeH)

        self.ldp_prior = ldp_prior

        self.n_ldparams = n_ldparams



    def get_sol(self,*args,**kwargs):
        """
        reads the guess transit fit solution

        There are 6 args for every planet
        T0, period, impact paramter, rp/rs, ecosw and esinw

        optional keywords, these are kept fixed (for now)
        dil : float, optional
            dilution
        veloffset : float, optional
            velocity zeropoint
        rvamp : float, optional
            radial velocity amplitude from doppler beaming
        occ : float, optional
            occultation depth
        ell : float, optional
            amplitude of ellipsoidal variations
        alb : float, optional
            geometric albedo of the planet
        """

        assert len(args) == self.nplanets * 6

        if 'dil' in kwargs.keys():
            dil = kwargs['dil']
            print ' running with dil = %s' %(dil)
        else:
            dil = 0.0
        if 'veloffset' in kwargs.keys():
            veloffset = kwargs['veloffset']
        else:
            veloffset = 0.0
        if 'rvamp' in kwargs.keys():
            rvamp = kwargs['rvamp']
        else:
            rvamp = 0.0
        if 'occ' in kwargs.keys():
            occ = kwargs['occ']
        else:
            occ = 0.0
        if 'ell' in kwargs.keys():
            ell = kwargs['ell']
        else:
            ell = 0.0
        if 'alb' in kwargs.keys():
            alb = kwargs['alb']
        else:
            alb = 0.0

        try:
            if self.zpt_0 == 0.:
                self.zpt_0 = 1.E-10
        except AttributeError:
            self.zpt_0 = 1.E-10
            self.zpt_0_unc = 1.E-6

        if self.n_ldparams == 2:
            fit_sol = np.array([self.rho_0,self.zpt_0,
                self.ld1,self.ld2])
        elif self.n_ldparams == 4:
            fit_sol = np.array([self.rho_0,self.zpt_0,
                self.ld1,self.ld2,self.ld3, self.ld4])
        for i in xrange(self.nplanets):
            T0_0 = args[i*6]
            per_0 = args[i*6 +1]
            b_0 = args[i*6 +2]
            rprs_0 = args[i*6 +3]
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
        zpt_unc = 1.E-8
        ld1_unc = 0.05
        ld2_unc = 0.05
        ld3_unc = 0.05
        ld4_unc = 0.05
        T0_unc = 0.0002
        per_unc = 0.00005
        b_unc = 0.001
        rprs_unc = 0.0001
        ecosw_unc = 0.001
        esinw_unc = 0.001
        #p0 = np.zeros([nwalkers,4+self.nplanets*6])
        if self.n_ldparams == 2:
            p0 = np.zeros([nwalkers,4+self.nplanets*6+1])
        elif self.n_ldparams == 4:
            p0 = np.zeros([nwalkers,6+self.nplanets*6+1])

        rho = self.fit_sol[0]
        zpt = self.fit_sol[1]
        ld1 = self.fit_sol[2]
        ld2 = self.fit_sol[3]

        if self.n_ldparams == 4:
            ld3 = self.fit_sol[4]
            ld4 = self.fit_sol[5]
            addval = 2

            start,stop = (0.0 - ld3) / ld3_unc, (1.0 - ld3) / ld3_unc
            p0[...,4] = truncnorm.rvs(start,stop
                ,loc=ld3,scale=ld3_unc,size=nwalkers)

            start,stop = (0.0 - ld4) / ld4_unc, (1.0 - ld4) / ld4_unc
            p0[...,5] = truncnorm.rvs(start,stop
                ,loc=ld4,scale=ld4_unc,size=nwalkers)

        else:
            addval = 0

        start,stop = (0.0001 - rho) / rho_unc, (30.0 - rho) / rho_unc
        p0[...,0] = truncnorm.rvs(start,stop
                ,loc=rho,scale=rho_unc,size=nwalkers)

        p0[...,1] = np.random.normal(loc=zpt,scale=zpt,size=nwalkers)

        start,stop = (0.0 - ld1) / ld1_unc, (1.0 - ld1) / ld1_unc
        p0[...,2] = truncnorm.rvs(start,stop
                ,loc=ld1,scale=ld1_unc,size=nwalkers)

        start,stop = (0.0 - ld2) / ld2_unc, (1.0 - ld2) / ld2_unc
        p0[...,3] = truncnorm.rvs(start,stop
                ,loc=ld2,scale=ld2_unc,size=nwalkers)


        for i in xrange(self.nplanets):
            (T0,per,b,rprs,ecosw,
                esinw) = self.fit_sol[i*6+4+addval:i*6 + 10+addval]
            b = 0.2
            ecosw = 0.0
            esinw = 0.0
            p0[...,i*6+4+addval] = np.random.normal(
                T0,T0_unc,size=nwalkers)
            p0[...,i*6+5+addval] = np.random.normal(
                per,per_unc,size=nwalkers)
            start,stop = (0.0 - b) / b_unc, (0.5 - b) / b_unc
            p0[...,i*6+6+addval] = truncnorm.rvs(
                start,stop
                ,loc=b,scale=b_unc,size=nwalkers)
            start,stop = (0.0 - rprs) / rprs_unc, (0.5 - rprs) / rprs_unc
            p0[...,i*6+7+addval] = truncnorm.rvs(
                start,stop
                ,loc=rprs,scale=rprs_unc,size=nwalkers)
            start,stop = (0.0 - ecosw) / ecosw_unc, (0.5 - ecosw) / ecosw_unc
            p0[...,i*6+8+addval] = truncnorm.rvs(
                start,stop
                ,loc=ecosw,scale=ecosw_unc,size=nwalkers)
            start,stop = (0.0 - esinw) / esinw_unc, (0.5 - esinw) / esinw_unc
            p0[...,i*6+9+addval] = truncnorm.rvs(
                start,stop
                ,loc=esinw,scale=esinw_unc,size=nwalkers)

        #this is the jitter term
        #make it like self.err
        errterm = np.median(self.err)
        start,stop = 0.0,10.
        p0[...,-1] = truncnorm.rvs(start,stop,
            loc=0.0,scale=0.1*errterm,size=nwalkers)
        return p0

    def cut_non_transit(self,ntdur=10):
        #make a mask for each planet candidate
        self.onlytransits = True
        tregion = np.zeros(self.nplanets)

        maskdat = np.zeros([self.npt,self.nplanets],dtype=bool)

        if self.n_ldparams == 2:
            addval = 0
        elif self.n_ldparams == 4:
            addval = 2
        for i in xrange(self.nplanets):
            T0 = self.fit_sol[i*6 + 4+addval]
            per = self.fit_sol[i*6 + 5+addval]
            rho = self.fit_sol[0]

            ars = self.get_ar(rho,per)
            tdur_dys = (1./ars) * per * (1./np.pi)

            #this is buggy because T0 is not nessessarily time of first transit
            #but time of a transit. So fudge.
            #subtract make T0 the first transit
            time0 = np.copy(T0)
            while True:
                if time0 - per < self.time[0]:
                    break
                else:
                    time0 = time0 - per

            ntransits = int((self.time[-1] - self.time[0]) / per) + 1
            t_times = np.arange(ntransits)*per + T0

            #make sure the first and last transit are not excluded even if
            #partially in the data
            t_times = np.r_[t_times,t_times[0] - per,t_times[-1] + per]

            for j in t_times:
                maskdat[:,i] = np.logical_or(maskdat[:,i],
                    np.logical_and(
                    self.time < j +tdur_dys*ntdur,
                    self.time > j - tdur_dys*ntdur) )
            tregion[i] = ntdur*tdur_dys
        #create a final mask that is the OR of the
        #individual masks
        finmask = np.zeros(self.npt)
        for i in xrange(self.nplanets):
            finmask = np.logical_or(finmask,maskdat[:,i])

        self.time = self.time[finmask]
        self.flux = self.flux[finmask]
        self.err = self.err[finmask]
        self._itime = self._itime[finmask]
        self._datatype = self._datatype[finmask]
        self.tregion = tregion



def get_ar(rho,period):
        """ gets a/R* from period and mean stellar density"""
        G = 6.67E-11
        rho_SI = rho * 1000.
        tpi = 3. * np.pi
        period_s = period * 86400.
        part1 = period_s**2 * G * rho_SI
        ar = (part1 / tpi)**(1./3.)
        return ar

def logchi2(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    onlytransits=False,tregion=0.0):
        #here are some priors to keep values sensible
        rho = fitsol[0]
        if rho < 0.0001 or rho > 100.:
            return -np.inf
        rprs = fitsol[np.arange(nplanets)*6 + 5]
        if np.any(rprs < 0.) or np.any(rprs > 0.5):
            return -np.inf
        ecosw = fitsol[np.arange(nplanets)*6 + 6]
        if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
            return -np.inf
        esinw = fitsol[np.arange(nplanets)*6 + 7]
        if np.any(esinw < -1.0) or np.any(esinw > 1.0):
            return -np.inf
        #avoid parabolic orbits
        ecc = np.sqrt(esinw**2 + ecosw**2)
        if np.any(ecc > 1.0):
            return -np.inf
        #avoid orbits where the planet enters the star
        per = fitsol[np.arange(nplanets)*6 + 3]
        ar = get_ar(rho,per)
        if np.any(ecc > (1.-(1./ar))):
            return -np.inf

        b = fitsol[np.arange(nplanets)*6 + 4]
        if np.any(b < 0.) or np.any(b > 1.0 + rprs):
            return -np.inf
        if onlytransits:
            T0 = fitsol[np.arange(nplanets)*6 + 2]
            if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
                return -np.inf

        model_lc = calc_model(fitsol,nplanets,fixed_sol,
            time,itime,ntt,tobs,omc,datatype)

        if rho_prior:
            chi2prior = (rho_0 - rho)**2 / rho_0_unc**2
        else:
            chi2prior = 0.0


        ecc[ecc == 0.0] = 1.E-10
        chi2ecc = np.log(1. / ecc)

        chi2val = np.sum((model_lc - flux)**2 / err**2)
        chi2const = np.log(1. / (np.sqrt(2.*np.pi) * np.mean(err)))
        chi2tot = (-chi2val/2.) + chi2prior
        #include eccentricity in the prior
        #having np.log(chi2ecc) -> e**(-chi2/2) / ecc
        logp = chi2tot + np.sum(chi2ecc)
        return logp


def calc_model(fitsol,nplanets,fixed_sol,time,itime,ntt,tobs,omc,datatype):
    sol = np.zeros([8 + 10*nplanets])
    rho = fitsol[0]
    zpt = fitsol[1]
    ld1,ld2,ld3,ld4 = fixed_sol[0:4]
    dil = fixed_sol[4]
    veloffset = fixed_sol[5]

    fixed_stuff = fixed_sol[6:10]

    sol[0:8] = np.array([rho,ld1,ld2,ld3,ld4,
        dil,veloffset,zpt])
    for i in xrange(nplanets):
        sol[8+(i*10):8+(i*10)+10] = np.r_[fitsol[2+i*6:8+i*6],fixed_stuff]

    tmodout = tmod.transitmodel(nplanets,sol,time,itime,
        ntt,tobs,omc,datatype)

    return tmodout - 1.

def logchi2_paramprior(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    teff_0,teff_0_unc,logg_0,logg_0_unc,feh_0,feh_0_unc,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/',
    onlytransits=False,tregion=0.0):
        minf = -np.inf
        #here are some priors to keep values sensible
        rho = fitsol[0]
        if rho < 1.E-6 or rho > 100.:
            return minf

        teff = fitsol[2]
        if teff < 3500 or teff > 50000.:
            return minf

        logg = fitsol[3]
        if logg < 0.0 or logg > 5.:
            return minf

        feh = fitsol[4]
        if feh < -5. or feh > 1.:
            return minf

        rprs = fitsol[np.arange(nplanets)*6 + 8]
        if np.any(rprs < 0.) or np.any(rprs > 0.5):
            return minf

        ecosw = fitsol[np.arange(nplanets)*6 + 9]
        if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
            return minf

        esinw = fitsol[np.arange(nplanets)*6 + 10]
        if np.any(esinw < -1.0) or np.any(esinw > 1.0):
            return minf

        #avoid parabolic orbits
        ecc = np.sqrt(esinw**2 + ecosw**2)
        if np.any(ecc > 1.0):
            return minf

        #avoid orbits where the planet enters the star
        per = fitsol[np.arange(nplanets)*6 + 6]
        ar = get_ar(rho,per)
        if np.any(ecc > (1.-(1./ar))):
            return minf

        b = fitsol[np.arange(nplanets)*6 + 7]
        if np.any(b < 0.) or np.any(b > 1.0 + rprs):
            return minf

        if onlytransits:
            T0 = fitsol[np.arange(nplanets)*6 + 5]
            if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
                return minf

        #calc thing limb darkening here
        if n_ldparams == 2:
            #if teff < 3500 and logg >= 3.5:
            if False:
                #this block should never run
                ldfile = ldfileloc + 'claret-quad-phoenix.txt'
                ld1,ld2 = ld_quad(ldfile,
                    teff,logg)
                ld3 = 0.0
                ld4 = 0.0
            #elif logg < 3.5 or teff >= 3500:
            if True:
                ldfile = ldfileloc + 'claret-limb-quad.txt'
                ld1,ld2 = claretquad(ldfile,
                    teff,logg,feh)
                ld3 = 0.0
                ld4 = 0.0
        elif n_ldparams == 4:
            ldfile = ldfileloc + 'claret-limb.txt'
            ld1,ld2,ld3,ld4 = claretlimb4p(ldfile,
                teff,logg,feh)

        lds = np.array([ld1,ld2,ld3,ld4])


        fitsol_model_calc = np.r_[fitsol[0:2],fitsol[5:]]
        fixed_sol_model_calc = np.r_[lds,fixed_sol]

        model_lc = calc_model(fitsol_model_calc,nplanets,fixed_sol_model_calc,
            time,itime,ntt,tobs,omc,datatype)

        if rho_prior:
            rho_prior = (rho_0 - rho)**2 / rho_0_unc**2
            #teff_prior = (teff_0 - teff)**2 / teff_0_unc**2
            #logg_prior = (logg_0 - logg)**2 / logg_0_unc**2
            #feh_prior = (feh_0 - feh)**2 / feh_0_unc**2
            #chi2prior = rho_prior+teff_prior+logg_prior+feh_prior

        else:
            rho_prior = 0.0

        teff_prior = (teff_0 - teff)**2 / teff_0_unc**2
        logg_prior = (logg_0 - logg)**2 / logg_0_unc**2
        feh_prior = (feh_0 - feh)**2 / feh_0_unc**2
        chi2prior = -0.5*(rho_prior+teff_prior+logg_prior+feh_prior)

        ecc[ecc == 0.0] = 1.E-10
        chi2ecc = np.log(1. / ecc)

        chi2val = -0.5*np.sum(((model_lc - flux)* (model_lc - flux))
            / (err*err))
        #chi2const = np.log(np.sum(1./(np.sqrt(2.*np.pi)*err)))
        chi2const = 0.0
        chi2tot = chi2const + chi2val + chi2prior
        #include eccentricity in the prior
        #having np.log(chi2ecc) -> e**(-chi2/2) / ecc
        logp = chi2tot + np.sum(chi2ecc)
        return logp

def logchi2_fitldp(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    ld1_0,ld1_0_unc,ld2_0,ld2_0_unc,ldp_prior,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/',
    onlytransits=False,tregion=0.0):
        minf = -np.inf
        #here are some priors to keep values sensible
        rho = fitsol[0]
        if rho < 1.E-6 or rho > 100.:
            return minf

        ld1 = fitsol[2]
        ld2 = fitsol[3]
        #some lind darkening constraints
        #from Burke et al. 2008 (XO-2b)
        if ld1 < 0.0:
            return minf
        if ld1 + ld2 > 1.0:
            return minf
        if ld1 + 2.*ld2 < 0.0:
            return minf
        if ld2 < -0.8:
            return minf


        if n_ldparams == 2:
            ld3, ld4 = 0.0,0.0
            addval = 0
        elif n_ldparams == 4:
            ld3 = fitsol[4]
            ld4 = fitsol[5]
            addval = 2

        rprs = fitsol[np.arange(nplanets)*6 + 7 + addval]
        if np.any(rprs < 0.) or np.any(rprs > 0.5):
            return minf

        ecosw = fitsol[np.arange(nplanets)*6 + 8+addval]
        if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
            return minf

        esinw = fitsol[np.arange(nplanets)*6 + 9+addval]
        if np.any(esinw < -1.0) or np.any(esinw > 1.0):
            return minf

        #avoid parabolic orbits
        ecc = np.sqrt(esinw**2 + ecosw**2)
        if np.any(ecc > 1.0):
            return minf

        #avoid orbits where the planet enters the star
        per = fitsol[np.arange(nplanets)*6 + 5+addval]
        ar = get_ar(rho,per)
        if np.any(ecc > (1.-(1./ar))):
            return minf

        b = fitsol[np.arange(nplanets)*6 + 6+addval]
        if np.any(b < 0.) or np.any(b > 1.0 + rprs):
            return minf

        if onlytransits:
            T0 = fitsol[np.arange(nplanets)*6 + 4+addval]
            if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
                return minf

        jitter = fitsol[-1]
        if jitter < 0.0:
            return minf
        err_jit = np.sqrt(err**2 + jitter**2)
        err_jit2 = err**2 + jitter**2

        lds = np.array([ld1,ld2,ld3,ld4])

        fitsol_model_calc = np.r_[fitsol[0:2],fitsol[4:]]
        fixed_sol_model_calc = np.r_[lds,fixed_sol]

        model_lc = calc_model(fitsol_model_calc,nplanets,fixed_sol_model_calc,
            time,itime,ntt,tobs,omc,datatype)

        # if rho_prior:
        #     rhoprior = (rho_0 - rho)**2 / rho_0_unc**2

        # else:
        #     rhoprior = 0.0



        # if ldp_prior:
        #     ldprior1 = (ld1_0 - ld1)*(ld1_0 - ld1) / ld1_0_unc**2
        #     ldprior2 = (ld2_0 - ld2)*(ld2_0 - ld2) / ld2_0_unc**2
        #     ldprior = ldprior1 + ldprior2
        # else:
        #     ldprior = 0.0

        # chi2prior = -0.5*(rhoprior+ldprior)

        ecc[ecc == 0.0] = 1.E-10
        #chi2ecc = np.log(1. / ecc)

        # chi2val = -0.5*np.sum(((model_lc - flux)* (model_lc - flux))
        #     / (err_jit*err_jit))
        # chi2const = -1.0*np.sum(np.log(err_jit))
        # #chi2const = 0.0
        # chi2tot = chi2const + chi2val + chi2prior
        # #include eccentricity in the prior
        # #having np.log(chi2ecc) -> e**(-chi2/2) / ecc
        # logp = chi2tot + np.sum(chi2ecc)

        npt_lc = len(err_jit)


        loglc = (
            - (npt_lc/2.)*np.log(2.*np.pi)
            - 0.5 * np.sum(np.log(err_jit2))
            - 0.5 * np.sum((model_lc - flux)**2 / err_jit2)
            )

        if rho_prior:
            logrho = (
                - 0.5 * np.log(2.*np.pi)
                - 0.5 * np.log(rho_0_unc**2)
                - 0.5 * (rho_0 - rho)**2 / rho_0_unc**2
                )
        else:
            rho_prior = 0.0

        if ldp_prior:
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
        else:
            logldp = 0.0

        logecc = - np.sum(np.log(ecc))

        logLtot = loglc + logrho + logldp + logecc

        return logLtot

# def calc_model_paramprior(fitsol,nplanets,fixed_sol,time,itime,ntt,tobs,omc,datatype):
#     sol = np.zeros([8 + 10*nplanets])
#     rho = fitsol[0]
#     zpt = fitsol[1]
#     ld1,ld2,ld3,ld4 = fixed_sol[0:4]
#     dil = fixed_sol[4]
#     veloffset = fixed_sol[5]

#     fixed_stuff = fixed_sol[6:10]

#     sol[0:8] = np.array([rho,ld1,ld2,ld3,ld4,
#         dil,veloffset,zpt])
#     for i in xrange(nplanets):
#         sol[8+(i*10):8+(i*10)+10] = np.r_[fitsol[2+i*6:8+i*6],fixed_stuff]

#     tmodout = tmod.transitmodel(nplanets,sol,time,itime,
#         ntt,tobs,omc,datatype)

#     return tmodout - 1.

def get_stats(par_arr,noprint=False):
    par_arr
    onesig = (1. - 0.682689492) / 2.
    twosig = (1. - 0.954499736) / 2.
    threesig = (1. - 0.997300204) / 2.
    med = np.median(par_arr)
    stdev = np.std(par_arr)
    sort_arr = np.sort(par_arr)
    nval = len(par_arr)
    m1 = med - sort_arr[np.floor(onesig * nval)]
    p1 = sort_arr[np.floor(nval - (onesig * nval))] - med
    m2 = med - sort_arr[np.floor(twosig * nval)]
    p2 = sort_arr[np.floor(nval - (twosig * nval))] - med
    m3 = med - sort_arr[np.floor(threesig * nval)]
    p3 = sort_arr[np.floor(nval - (threesig * nval))] - med
    ninefivelow = sort_arr[np.floor(0.025*nval)]
    ninefivehigh = sort_arr[np.floor(0.975*nval)]
    if not noprint:
        print '95percent credible interval = %s - %s' %(ninefivelow,ninefivehigh)
    return np.array([med,stdev,p1,m1,p2,m2,p3,m3])

def model_real_paramprior(rho,zpt,teff,logg,feh,T0,
    per,b,rprs,ecosw,esinw,
    time,itime,ntt,tobs,omc,datatype,
    n_ldparams=2,
    ldfileloc='/Users/tom/svn_code/tom_code/'):
    ldfile = ldfileloc + 'claret-limb-quad.txt'
    ld1,ld2 = claretquad(ldfile,teff,logg,feh)
    ld3 = 0.0
    ld4 = 0.0

    dil=0.0
    veloffset = 0.0

    rvamp = 0.0
    occ = 0.0
    ell = 0.0
    alb = 0.0

    nplanets = 1

    sol = np.array([rho,ld1,ld2,ld3,ld4,
        dil,veloffset,zpt,T0,per,b,rprs,ecosw,esinw,
        rvamp,occ,ell,alb])

    tmodout = tmod.transitmodel(nplanets,sol,time,itime,
        ntt,tobs,omc,datatype) - 1.0

    return tmodout



def testtom(t,num):
    rho,zpt,teff,logg,feh,T0,per,b,rprs,ecosw,esinw = (t[...,num])
    mod = model_real_paramprior(rho,zpt,teff,logg,feh,T0,per,b,rprs,ecosw,
        esinw,M.time,M._itime,M._ntt,M._tobs,M._omc,M._datatype,
        n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/')
    q,f = get_qf(M.time,a,per,T0)
    plt.plot(q,f,alpha=0.5)

def run_crap(t):
    for num in random.choice(np.arange(len(t[1])),size=10):
        testtom(t,num)
    q,f = get_qf(M.time,M.flux,per,T0)
    plt.scatter(q,f,s=1,color='k',alpha=0.2)



def get_qf(time,flux,period,epoch):
    date1 = (time - epoch) + 0.5*period
    phi1 = (((date1 / period) - np.floor(date1/period)) * 24. * period) - 12*period

    q1 = np.sort(phi1)
    f1 = (flux[np.argsort(phi1)]) * 1.E6
    return q1, f1




