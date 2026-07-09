import numpy as np
import scipy.special as sysp
import scipy.spatial as syspat
import pandas as pd
import gc
import time

#///////////////////////////////////////////////////////////////////
# HOD Fits
#///////////////////////////////////////////////////////////////////
class HODFits(object):
    def __init__(self):
        self.bf = {'lgMmin':np.array([12.33,-0.85,0.19]),
                   'siglgM':np.array([0.44,-0.16,0.3]),
                   'lgM1':np.array([13.52,-0.72,0.16]),
                   'lgM0':np.array([12.24,-0.54,0.0]),
                   'alpha':np.array([1.16,-0.20,0.10])}

    def hod_fit(self,param,Mr):
        a0,a1,a2 = self.bf[param]
        x = Mr + 20.5
        out = (a0 + a1*x + a2*x**2
               if param != 'siglgM' else
               a0 + a1*sysp.erf(x/a2))
        return out

    def hod_fit_deriv(self,param,Mr):
        a0,a1,a2 = self.bf[param]
        x = Mr + 20.5
        out = (a1 + 2*a2*x
               if param != 'siglgM' else
               (a1/a2)*2/np.sqrt(np.pi)*np.exp(-(x/a2)**2))
        return out

#///////////////////////////////////////////////////////////////////
# HOD Functions
#///////////////////////////////////////////////////////////////////
class HODFunctions(HODFits):
    def __init__(self):
        HODFits.__init__(self)

    def erf_arg(self,Mr,lgm):
        Mr_sc = np.isscalar(Mr)
        lgm_sc = np.isscalar(lgm)
        lgm_arr = np.outer(lgm,np.ones(Mr.size)) if ((not Mr_sc) & (not lgm_sc)) else 1.0*lgm
        erfarg = (lgm_arr - self.hod_fit('lgMmin',Mr))/self.hod_fit('siglgM',Mr)
        return erfarg

    def fcen_thresh(self,Mr,lgm):
        erfarg = self.erf_arg(Mr,lgm)
        out = 0.5*(1+sysp.erf(erfarg))
        return out

    def Nsat_thresh(self,Mr,lgm):
        M0 = 10**self.hod_fit('lgM0',Mr)
        M1 = 10**self.hod_fit('lgM1',Mr)
        alpha = self.hod_fit('alpha',Mr)
        mhalo = 10**lgm
        Mr_sc = np.isscalar(Mr)
        lgm_sc = np.isscalar(lgm)
        if Mr_sc & lgm_sc:
            out = ((mhalo - M0)/M1)**alpha if mhalo > M0 else 0.0
        else:
            mhalo_arr = np.outer(mhalo,np.ones(Mr.size)) if ((not Mr_sc) & (not lgm_sc)) else 1.0*mhalo
            Dm = (mhalo_arr - M0)/M1
            Dm[Dm < 0.0] = 0.0
            out = Dm**alpha
        return out

#///////////////////////////////////////////////////////////////////
# Mocker Baseline (Original loop-based implementation)
#///////////////////////////////////////////////////////////////////
class MockerOld(HODFunctions):
    def __init__(self, synthetic_halos):
        HODFunctions.__init__(self)
        self.synthetic_halos = synthetic_halos
        self.redshift = 0.0
        self.Om = 0.276
        self.Lbox = 300.0
        self.massdef = 'm200b'
        self.mmin = 2e11
        self.Mrmax = -20.5
        self.Dvir = 200*self.Om
        self.rhoc = 2.7754e11
        self.rhoc_z = self.rhoc*self.EHub(self.redshift)**2
        self.rng = np.random.RandomState(seed=42)
        dMr = 0.001
        nMr = int((23.5+self.Mrmax)/dMr)
        self.Mrvals = np.linspace(-23.5,self.Mrmax,nMr)
        self.dMr = self.Mrvals[1]-self.Mrvals[0]
        
    def EHub(self,z):
        return np.sqrt(self.Om*(1+z)**3 + 1-self.Om)
    
    def read_this(self):
        return self.synthetic_halos
    
    def prep_halo_data(self):
        halos = self.read_this()
        Nhalos_all = halos.size
        cond_clean = (halos[self.massdef] >= self.mmin)
        cond_clean = cond_clean & (halos['pid'] == -1)
        return halos[cond_clean]

    def occupy_halos(self,halo_mass):
        occupy = np.ones(halo_mass.size,dtype=bool)
        fcen_min = self.fcen_thresh(self.Mrmax,np.log10(halo_mass))
        u = self.rng.rand(halo_mass.size)
        occupy[u > fcen_min] = False
        return occupy

    def assign_centrals(self,halos):
        centrals = np.zeros(halos.size,dtype=[('haloid','int'),
                                              ('lgm','double'),('rvir','double'),('con','double'),
                                              ('x','double'),('y','double'),('z','double'),
                                              ('Mr','double'),('Nsat','int')])
        centrals['haloid'] = halos['ID']
        centrals['lgm'] = np.log10(halos[self.massdef])
        centrals['x'] = halos['x']
        centrals['y'] = halos['y']
        centrals['z'] = halos['z']
        
        rvir = (3*halos[self.massdef]/(4*np.pi*self.Dvir*self.rhoc_z))**(1/3.)
        centrals['rvir'] = rvir*(1+self.redshift)
        centrals['con'] = rvir/(halos['rs']*1e-3/(1+self.redshift))
        centrals['Mr'] = self.assign_central_luminosity(centrals['lgm'])
        centrals['Nsat'] = self.get_Poisson(centrals['lgm'])
        return centrals

    def get_Poisson(self,lgm_halos):
        mean_Nsat = self.Nsat_thresh(self.Mrmax,lgm_halos)
        Nsat_values = self.rng.poisson(mean_Nsat)
        return Nsat_values
    
    def assign_central_luminosity(self,lgm_halos):
        Mr_cen = np.zeros(lgm_halos.size,dtype=float)
        u = self.rng.rand(lgm_halos.size)
        for h in range(lgm_halos.size):
            PcenL = self.fcen_thresh(self.Mrvals,lgm_halos[h])/self.fcen_thresh(self.Mrmax,lgm_halos[h])
            Mr_upp = self.Mrvals[PcenL > u[h]][0]
            Mr_low = self.Mrvals[PcenL <= u[h]][-1]
            Mr_cen[h] = np.max([Mr_upp,Mr_low])
        return Mr_cen

    def assign_satellites(self,centrals):
        Nsat_tot = np.sum(centrals['Nsat'])
        satellites = np.zeros(Nsat_tot,dtype=[('haloid','int'),
                                              ('lgm','double'),('con','double'),
                                              ('x','double'),('y','double'),('z','double'),
                                              ('Mr','double')])
        s_lo = 0
        for h in range(centrals.size):
            s_hi = s_lo + centrals['Nsat'][h]
            if centrals['Nsat'][h]:
                sl = np.s_[s_lo:s_hi]
                satellites['haloid'][sl] = centrals['haloid'][h]
                satellites['lgm'][sl] = centrals['lgm'][h]
                satellites['con'][sl] = centrals['con'][h]
            s_lo = s_hi

        s_lo = 0
        for h in range(centrals.size):
            s_hi = s_lo + centrals['Nsat'][h]
            if centrals['Nsat'][h]:
                sl = np.s_[s_lo:s_hi]
                sx,sy,sz = self.assign_satellite_pos(centrals,h)
                satellites['x'][sl] = sx
                satellites['y'][sl] = sy
                satellites['z'][sl] = sz
            s_lo = s_hi

        s_lo = 0
        for h in range(centrals.size):
            s_hi = s_lo + centrals['Nsat'][h]
            if centrals['Nsat'][h]:
                sMr = self.assign_satellite_luminosities(centrals,h)
                satellites['Mr'][s_lo:s_hi] = sMr
            s_lo = s_hi
        return satellites
    
    def assign_satellite_luminosities(self,centrals,h):
        Nsat_h = centrals['Nsat'][h]
        lgmhalo = centrals['lgm'][h]
        PsatL = self.Nsat_thresh(self.Mrvals,lgmhalo)/self.Nsat_thresh(self.Mrmax,lgmhalo)
        u = self.rng.rand(Nsat_h)
        Mr_sat = np.array([self.Mrvals[(PsatL >= u[s])][0] for s in range(Nsat_h)])
        return Mr_sat
    
    def assign_satellite_pos(self,centrals,h):
        Nsat_h = centrals['Nsat'][h]
        rvir = centrals['rvir'][h]
        con = centrals['con'][h]
        sat_pos = self.gen_NFW_profile(Nsat_h,cvir=con,Rvir=rvir,rng=self.rng)
        sx = sat_pos[:,0] + centrals['x'][h]
        sy = sat_pos[:,1] + centrals['y'][h]
        sz = sat_pos[:,2] + centrals['z'][h]
        sx = sx % self.Lbox
        sy = sy % self.Lbox
        sz = sz % self.Lbox
        return sx,sy,sz
    
    def gen_NFW_profile(self,Nsat,cvir=None,Rvir=None,rng=None,include_cen=False,clean_up=False):
        if rng is None:
            rng = np.random.RandomState(42)
        phi = 2*np.pi*rng.rand(Nsat)
        cos_theta = 2*rng.rand(Nsat) - 1
        sin_theta = np.sqrt(1-cos_theta**2)
        rsamp = self.gen_rsamp(Nsat,cvir=cvir,Rvir=Rvir,rng=rng,clean_up=clean_up)
        x_trc = rsamp*sin_theta*np.sin(phi)
        y_trc = rsamp*sin_theta*np.cos(phi)
        z_trc = rsamp*cos_theta
        sample = np.array([x_trc,y_trc,z_trc]).T
        if include_cen:
            sample = np.append(sample,[[0.,0.,0.]],axis=0)
        return sample
    
    def gen_rsamp(self,Nsat,cvir=None,Rvir=None,rng=None,clean_up=False):
        xmax = 2*cvir
        if rng is None:
            rng = np.random.RandomState(42)
        xfine = np.linspace(0,xmax,100000)
        dx = xfine[1]-xfine[0]
        rs = Rvir/cvir
        Px = self.Mencl_nfw(xfine)
        ind_small = np.where(xfine < 1e-3)[0]
        if ind_small.size:
            Px[ind_small] = xfine[ind_small]**2/2 - 2*xfine[ind_small]**3/3.0 
            Px[ind_small] = Px[ind_small] + 3*xfine[ind_small]**4/4.0 - 4*xfine[ind_small]**5/5.0
        Px /= self.Mencl_nfw(cvir)
        vran = rng.rand(Nsat)
        rsamp = np.interp(vran,Px,xfine)
        rsamp *= rs
        return rsamp
    
    def Mencl_nfw(self,x):
        return np.log(1+x) - x/(1+x)

    def mock_it(self):
        halos = self.prep_halo_data()
        occupy = self.occupy_halos(halos[self.massdef])
        halos = halos[occupy]
        all_masses = halos[self.massdef]
        centrals = self.assign_centrals(halos)
        satellites = self.assign_satellites(centrals)
        return centrals, satellites, all_masses

#///////////////////////////////////////////////////////////////////
# Mocker New Base (Vectorized)
#///////////////////////////////////////////////////////////////////
class MockerNewBase(HODFunctions):
    def __init__(self, synthetic_halos, chunk_size=10000):
        HODFunctions.__init__(self)
        self.synthetic_halos = synthetic_halos
        self.chunk_size = chunk_size
        self.redshift = 0.0
        self.Om = 0.276
        self.Lbox = 300.0
        self.massdef = 'm200b'
        self.mmin = 2e11
        self.Mrmax = -20.5
        self.Dvir = 200*self.Om
        self.rhoc = 2.7754e11
        self.rhoc_z = self.rhoc*self.EHub(self.redshift)**2
        self.rng = np.random.RandomState(seed=42)
        dMr = 0.001
        nMr = int((23.5+self.Mrmax)/dMr)
        self.Mrvals = np.linspace(-23.5,self.Mrmax,nMr)
        self.dMr = self.Mrvals[1]-self.Mrvals[0]
        
        # Precompute HOD parameters for self.Mrvals and self.Mrmax
        self.lgMmin_Mrvals = self.hod_fit('lgMmin', self.Mrvals)
        self.siglgM_Mrvals = self.hod_fit('siglgM', self.Mrvals)
        self.M0_Mrvals = 10**self.hod_fit('lgM0', self.Mrvals)
        self.M1_Mrvals = 10**self.hod_fit('lgM1', self.Mrvals)
        self.alpha_Mrvals = self.hod_fit('alpha', self.Mrvals)

        self.lgMmin_Mrmax = self.hod_fit('lgMmin', self.Mrmax)
        self.siglgM_Mrmax = self.hod_fit('siglgM', self.Mrmax)
        self.M0_Mrmax = 10**self.hod_fit('lgM0', self.Mrmax)
        self.M1_Mrmax = 10**self.hod_fit('lgM1', self.Mrmax)
        self.alpha_Mrmax = self.hod_fit('alpha', self.Mrmax)

    def EHub(self,z):
        return np.sqrt(self.Om*(1+z)**3 + 1-self.Om)

    def erf_arg(self, Mr, lgm):
        if Mr is self.Mrvals:
            lgMmin = self.lgMmin_Mrvals
            siglgM = self.siglgM_Mrvals
            lgm_arr = lgm[:, np.newaxis]
        elif Mr == self.Mrmax:
            lgMmin = self.lgMmin_Mrmax
            siglgM = self.siglgM_Mrmax
            lgm_arr = lgm
        else:
            Mr_sc = np.isscalar(Mr)
            lgm_sc = np.isscalar(lgm)
            lgm_arr = lgm[:, np.newaxis] if ((not Mr_sc) and (not lgm_sc)) else 1.0 * lgm
            lgMmin = self.hod_fit('lgMmin', Mr)
            siglgM = self.hod_fit('siglgM', Mr)
        return (lgm_arr - lgMmin) / siglgM

    def fcen_thresh(self, Mr, lgm):
        erfarg = self.erf_arg(Mr, lgm)
        out = 0.5 * (1 + sysp.erf(erfarg))
        return out

    def Nsat_thresh(self, Mr, lgm):
        if Mr is self.Mrvals:
            M0 = self.M0_Mrvals
            M1 = self.M1_Mrvals
            alpha = self.alpha_Mrvals
            mhalo_arr = (10**lgm)[:, np.newaxis]
            Dm = (mhalo_arr - M0) / M1
            Dm[Dm < 0.0] = 0.0
            return Dm**alpha
        elif Mr == self.Mrmax:
            M0 = self.M0_Mrmax
            M1 = self.M1_Mrmax
            alpha = self.alpha_Mrmax
            mhalo = 10**lgm
            if np.isscalar(lgm):
                return ((mhalo - M0)/M1)**alpha if mhalo > M0 else 0.0
            else:
                Dm = (mhalo - M0) / M1
                Dm[Dm < 0.0] = 0.0
                return Dm**alpha
        else:
            M0 = 10**self.hod_fit('lgM0', Mr)
            M1 = 10**self.hod_fit('lgM1', Mr)
            alpha = self.hod_fit('alpha', Mr)
            mhalo = 10**lgm
            Mr_sc = np.isscalar(Mr)
            lgm_sc = np.isscalar(lgm)
            if Mr_sc and lgm_sc:
                out = ((mhalo - M0)/M1)**alpha if mhalo > M0 else 0.0
            else:
                mhalo_arr = mhalo[:, np.newaxis] if ((not Mr_sc) and (not lgm_sc)) else 1.0 * mhalo
                Dm = (mhalo_arr - M0) / M1
                Dm[Dm < 0.0] = 0.0
                out = Dm**alpha
            return out
    
    def read_this(self):
        return self.synthetic_halos
    
    def prep_halo_data(self):
        halos = self.read_this()
        cond_clean = (halos[self.massdef] >= self.mmin)
        cond_clean = cond_clean & (halos['pid'] == -1)
        return halos[cond_clean]

    def occupy_halos(self,halo_mass):
        occupy = np.ones(halo_mass.size,dtype=bool)
        fcen_min = self.fcen_thresh(self.Mrmax,np.log10(halo_mass))
        u = self.rng.rand(halo_mass.size)
        occupy[u > fcen_min] = False
        return occupy

    def assign_centrals(self,halos):
        centrals = np.zeros(halos.size,dtype=[('haloid','int'),
                                              ('lgm','double'),('rvir','double'),('con','double'),
                                              ('x','double'),('y','double'),('z','double'),
                                              ('Mr','double'),('Nsat','int')])
        centrals['haloid'] = halos['ID']
        centrals['lgm'] = np.log10(halos[self.massdef])
        centrals['x'] = halos['x']
        centrals['y'] = halos['y']
        centrals['z'] = halos['z']
        
        rvir = (3*halos[self.massdef]/(4*np.pi*self.Dvir*self.rhoc_z))**(1/3.)
        centrals['rvir'] = rvir*(1+self.redshift)
        centrals['con'] = rvir/(halos['rs']*1e-3/(1+self.redshift))
        centrals['Mr'] = self.assign_central_luminosity(centrals['lgm'])
        centrals['Nsat'] = self.get_Poisson(centrals['lgm'])
        return centrals

    def get_Poisson(self,lgm_halos):
        mean_Nsat = self.Nsat_thresh(self.Mrmax,lgm_halos)
        Nsat_values = self.rng.poisson(mean_Nsat)
        return Nsat_values
    
    def assign_central_luminosity(self, lgm_halos):
        Mr_cen = np.zeros(lgm_halos.size)
        u = self.rng.rand(lgm_halos.size)
        
        for i in range(0, lgm_halos.size, self.chunk_size):
            end = min(i + self.chunk_size, lgm_halos.size)
            lgm_chunk = lgm_halos[i:end]
            u_chunk = u[i:end]
            
            PcenL = self.fcen_thresh(self.Mrvals, lgm_chunk) / self.fcen_thresh(self.Mrmax, lgm_chunk)[:, np.newaxis]
            
            mask_gt = PcenL > u_chunk[:, np.newaxis]
            idx_upp = np.argmax(mask_gt, axis=1)
            Mr_upp = self.Mrvals[idx_upp]
            
            mask_le = PcenL <= u_chunk[:, np.newaxis]
            idx_low_reversed = np.argmax(mask_le[:, ::-1], axis=1)
            idx_low = PcenL.shape[1] - 1 - idx_low_reversed
            Mr_low = self.Mrvals[idx_low]
            
            any_le = np.any(mask_le, axis=1)
            Mr_low[~any_le] = -np.inf
            
            Mr_cen[i:end] = np.maximum(Mr_upp, Mr_low)

        return Mr_cen

    def assign_satellites(self, centrals):
        idx = np.repeat(np.arange(centrals.size), centrals['Nsat'])
        Nsat_tot = idx.size
        
        satellites = np.zeros(Nsat_tot, dtype=[('haloid','int'),
                                              ('lgm','double'),('con','double'),
                                              ('x','double'),('y','double'),('z','double'),
                                              ('Mr','double')])
        
        if Nsat_tot == 0:
            return satellites
            
        satellites['haloid'] = centrals['haloid'][idx]
        satellites['lgm'] = centrals['lgm'][idx]
        satellites['con'] = centrals['con'][idx]
        
        # Position assignment
        sat_pos = self.gen_NFW_profile_vectorized(centrals, idx)
        satellites['x'] = (sat_pos[:,0] + centrals['x'][idx]) % self.Lbox
        satellites['y'] = (sat_pos[:,1] + centrals['y'][idx]) % self.Lbox
        satellites['z'] = (sat_pos[:,2] + centrals['z'][idx]) % self.Lbox
        
        # Luminosities assignment
        satellites['Mr'] = self.assign_satellite_luminosities_vectorized(centrals['lgm'][idx])
        
        return satellites

    def assign_satellite_luminosities_vectorized(self, lgm_sat):
        Mr_sat = np.zeros(lgm_sat.size)
        u = self.rng.rand(lgm_sat.size)
        
        for i in range(0, lgm_sat.size, self.chunk_size):
            end = min(i + self.chunk_size, lgm_sat.size)
            lgm_chunk = lgm_sat[i:end]
            u_chunk = u[i:end]
            
            PsatL = self.Nsat_thresh(self.Mrvals, lgm_chunk) / self.Nsat_thresh(self.Mrmax, lgm_chunk)[:, np.newaxis]
            mask_ge = PsatL >= u_chunk[:, np.newaxis]
            idx_upp = np.argmax(mask_ge, axis=1)
            
            Mr_sat[i:end] = self.Mrvals[idx_upp]

        return Mr_sat

    def Mencl_nfw(self,x):
        return np.log(1+x) - x/(1+x)

    def mock_it(self):
        halos = self.prep_halo_data()
        occupy = self.occupy_halos(halos[self.massdef])
        halos = halos[occupy]
        all_masses = halos[self.massdef]
        centrals = self.assign_centrals(halos)
        satellites = self.assign_satellites(centrals)
        return centrals, satellites, all_masses


#///////////////////////////////////////////////////////////////////
# Path A Mocker: Loops over halos but vectorizes grid or reuses it
#///////////////////////////////////////////////////////////////////
class MockerNewPathA(MockerNewBase):
    def gen_NFW_profile_vectorized(self, centrals, idx):
        # Path A: exact same RNG draw order as original!
        # We loop over each halo, generate its random numbers, and construct/lookup the grid.
        # But wait! To optimize Path A, we can cache/reuse the xfine, Px grid if we round concentrations,
        # or build the grid dynamically. Let's see:
        # We want to match MockerOld EXACTLY.
        # Let's write the exact loop from MockerOld to make sure Path A has 100% exact match.
        # But we optimize by avoiding duplicate work if possible.
        # If we want exact RNG match, we must call self.rng.rand/uniform in the same sequence.
        # Since we use self.rng, the calls must be:
        # For each halo h with Nsat_h > 0:
        #   phi = 2*pi*rng.rand(Nsat_h)
        #   cos_theta = 2*rng.rand(Nsat_h) - 1
        #   vran = rng.rand(Nsat_h)
        # Yes! Let's check:
        # In MockerOld:
        #   gen_NFW_profile:
        #     phi = 2*np.pi*rng.rand(Nsat)
        #     cos_theta = 2*rng.rand(Nsat) - 1
        #     rsamp = self.gen_rsamp(Nsat, cvir=cvir, Rvir=Rvir, rng=rng)
        #       vran = rng.rand(Nsat)
        # So we draw: phi (size Nsat), cos_theta (size Nsat), then vran (size Nsat).
        # This exact order must be preserved!
        # Let's implement this loop. Can we make it faster than MockerOld?
        # In MockerOld, the main bottleneck is building xfine (size 100,000) every time.
        # Can we avoid rebuilding xfine if cvir is the same?
        # Actually, even if we rebuild it, can we optimize the grid size?
        # But wait, if we change grid size, np.interp changes slightly, so it wouldn't be a 100% exact match.
        # For exact match, we must keep size 100000.
        # Let's see: how many distinct cvir values are there?
        # If we cache the Mencl_nfw(xfine) and Taylor corrections, can we do it?
        # Yes! Note that Px = Mencl_nfw(xfine) / Mencl_nfw(cvir).
        # Since xfine = np.linspace(0, 2*cvir, 100000) = cvir * np.linspace(0, 2, 100000).
        # Let y = np.linspace(0, 2, 100000). This grid y is CONSTANT! It does not depend on cvir!
        # So xfine = cvir * y.
        # Let's check: Mencl_nfw(xfine) = np.log(1 + cvir * y) - (cvir * y) / (1 + cvir * y).
        # Wait, does Mencl_nfw(xfine) still depend on cvir? Yes.
        # But wait! We can compute it much faster:
        # 1. preallocate arrays.
        # 2. Since y is fixed, `ind_small` corresponds to `xfine < 1e-3` which is `y < 1e-3 / cvir`.
        # Let's look at the implementation of gen_rsamp in Path A:
        Nsat_tot = idx.size
        sat_pos = np.zeros((Nsat_tot, 3))
        
        # We only loop over halos with Nsat > 0
        s_lo = 0
        y_grid = np.linspace(0, 2, 100000)
        mencl_y_grid_cache = {} # We can cache if rounded cvir, but here we don't round for exact match
        
        # We can pre-compute some parts, but let's see how much speedup we get just by writing it clean.
        for h in range(centrals.size):
            Nsat_h = centrals['Nsat'][h]
            if Nsat_h == 0:
                continue
            
            rvir = centrals['rvir'][h]
            cvir = centrals['con'][h]
            
            # RNG draws in the exact order of MockerOld
            phi = 2 * np.pi * self.rng.rand(Nsat_h)
            cos_theta = 2 * self.rng.rand(Nsat_h) - 1
            sin_theta = np.sqrt(1 - cos_theta**2)
            vran = self.rng.rand(Nsat_h)
            
            # gen_rsamp logic
            xmax = 2 * cvir
            xfine = y_grid * cvir
            dx = xfine[1] - xfine[0]
            rs = rvir / cvir
            
            Px = np.log(1 + xfine) - xfine / (1 + xfine)
            ind_small = np.where(xfine < 1e-3)[0]
            if ind_small.size:
                # Taylor expansion for small x
                x_s = xfine[ind_small]
                Px[ind_small] = x_s**2/2 - 2*x_s**3/3.0 + 3*x_s**4/4.0 - 4*x_s**5/5.0
            
            Px /= (np.log(1 + cvir) - cvir / (1 + cvir))
            rsamp = np.interp(vran, Px, xfine)
            rsamp *= rs
            
            # Position calculation
            x_trc = rsamp * sin_theta * np.sin(phi)
            y_trc = rsamp * sin_theta * np.cos(phi)
            z_trc = rsamp * cos_theta
            
            sat_pos[s_lo : s_lo + Nsat_h] = np.column_stack((x_trc, y_trc, z_trc))
            s_lo += Nsat_h
            
        return sat_pos


#///////////////////////////////////////////////////////////////////
# Path B Mocker: Fully vectorized, batched RNG, not exact RNG match
#///////////////////////////////////////////////////////////////////
class MockerNewPathB(MockerNewBase):
    def gen_NFW_profile_vectorized(self, centrals, idx):
        # Path B: fully vectorized batch RNG and global interpolation!
        # No loop over halos.
        Nsat_tot = idx.size
        cvir_arr = centrals['con'][idx]
        Rvir_arr = centrals['rvir'][idx]
        
        phi = 2 * np.pi * self.rng.rand(Nsat_tot)
        cos_theta = 2 * self.rng.rand(Nsat_tot) - 1
        sin_theta = np.sqrt(1 - cos_theta**2)
        vran = self.rng.rand(Nsat_tot)
        
        # Build one master high-resolution NFW profile grid
        # We can set max_c to be the maximum concentration in our catalog
        max_c = np.max(cvir_arr)
        x_grid = np.logspace(-5, np.log10(max_c * 2.0), 100000)
        M_grid = np.log(1 + x_grid) - x_grid / (1 + x_grid)
        
        # Normalized target NFW mass percentiles for all satellites at once
        target_M = vran * (np.log(1 + cvir_arr) - cvir_arr / (1 + cvir_arr))
        
        # Find the normalized radius (x) for every satellite simultaneously
        x_samp = np.interp(target_M, M_grid, x_grid)
        rsamp = x_samp * (Rvir_arr / cvir_arr)
        
        x_trc = rsamp * sin_theta * np.sin(phi)
        y_trc = rsamp * sin_theta * np.cos(phi)
        z_trc = rsamp * cos_theta
        
        return np.column_stack((x_trc, y_trc, z_trc))


def make_synthetic_halos(n_halos=2000, seed=123):
    rng = np.random.RandomState(seed)
    halodatatype = {'ID':int,'descID':int,
                    'mbnd_vir':float,'vmax':float,'vrms':float,'rvir':float,'rs':float,'np':int,
                    'x':float,'y':float,'z':float,'vx':float,'vy':float,'vz':float,
                    'Jx':float,'Jy':float,'Jz':float,'spin':float,'rs_klypin':float,
                    'mvir':float,'m200b':float,'m200c':float,'mCustom2':float,'mCustom':float,
                    'Xoff':float,'Voff':float,'spin_bullock':float,'b_to_a':float,'c_to_a':float,
                    'Ax':float,'Ay':float,'Az':float,'b_to_a_500c':float,'c_to_a_500c':float,
                    'Ax_500c':float,'Ay_500c':float,'Az_500c':float,
                    'TbyU':float,'Mpe_Behroozi':float,'Mpe_Diemer':float,'halfmassradius':float,
                    'pid':int}
    dtype_list = list(halodatatype.items())
    halos = np.zeros(n_halos, dtype=dtype_list)
    halos['ID'] = np.arange(n_halos) + 1
    log_masses = rng.uniform(11.3, 14.8, n_halos)
    halos['m200b'] = 10**log_masses
    halos['x'] = rng.uniform(0, 300.0, n_halos)
    halos['y'] = rng.uniform(0, 300.0, n_halos)
    halos['z'] = rng.uniform(0, 300.0, n_halos)
    halos['pid'] = -1
    
    # Calculate rvir
    Dvir = 200 * 0.276
    rhoc_z = 2.7754e11
    rvir = (3 * halos['m200b'] / (4 * np.pi * Dvir * rhoc_z))**(1/3.0)
    con = rng.uniform(4.0, 20.0, n_halos)
    halos['rs'] = (rvir / con) * 1e3
    return halos

if __name__ == '__main__':
    print("Generating synthetic halos...")
    halos = make_synthetic_halos(n_halos=50000, seed=123)
    
    print("\n--- Running Baseline MockerOld ---")
    m_old = MockerOld(halos)
    t0 = time.time()
    centrals_old, satellites_old, masses_old = m_old.mock_it()
    t_old = time.time() - t0
    print(f"MockerOld took {t_old:.4f} seconds. Created {centrals_old.size} centrals, {satellites_old.size} satellites.")
    
    print("\n--- Running Vectorized MockerNewPathA (Exact Match) ---")
    m_new_a = MockerNewPathA(halos)
    t0 = time.time()
    centrals_new_a, satellites_new_a, masses_new_a = m_new_a.mock_it()
    t_new_a = time.time() - t0
    print(f"MockerNewPathA took {t_new_a:.4f} seconds. Created {centrals_new_a.size} centrals, {satellites_new_a.size} satellites.")
    
    print("\n--- Running Vectorized MockerNewPathB (Approx Match) ---")
    m_new_b = MockerNewPathB(halos)
    t0 = time.time()
    centrals_new_b, satellites_new_b, masses_new_b = m_new_b.mock_it()
    t_new_b = time.time() - t0
    print(f"MockerNewPathB took {t_new_b:.4f} seconds. Created {centrals_new_b.size} centrals, {satellites_new_b.size} satellites.")

    # 1. Verify occupation matching
    print("\n=== Verification ===")
    
    # Check centrals properties
    for prop in ['haloid', 'lgm', 'rvir', 'con', 'x', 'y', 'z', 'Mr', 'Nsat']:
        close = np.allclose(centrals_old[prop], centrals_new_a[prop])
        print(f"Centrals {prop:7s} match (Path A): {close}")
        if not close:
            diff = np.abs(centrals_old[prop] - centrals_new_a[prop])
            print(f"  Max diff: {np.max(diff)}")
            
    # Check satellites properties (Path A)
    for prop in ['haloid', 'lgm', 'con', 'x', 'y', 'z', 'Mr']:
        close = np.allclose(satellites_old[prop], satellites_new_a[prop])
        print(f"Satellites {prop:5s} match (Path A): {close}")
        if not close:
            diff = np.abs(satellites_old[prop] - satellites_new_a[prop])
            print(f"  Max diff: {np.max(diff)}")

    # Check satellites properties (Path B - coordinates and Mr distributions)
    print("\nPath B Distribution Comparison (should be statistically similar):")
    for prop in ['x', 'y', 'z']:
        mean_old, std_old = np.mean(satellites_old[prop]), np.std(satellites_old[prop])
        mean_new, std_new = np.mean(satellites_new_b[prop]), np.std(satellites_new_b[prop])
        print(f"Satellites {prop:5s} - Old: {mean_old:.4f} +/- {std_old:.4f} | New B: {mean_new:.4f} +/- {std_new:.4f}")
        
    print(f"\nSpeedup Path A: {t_old / t_new_a:.2f}x")
    print(f"Speedup Path B: {t_old / t_new_b:.2f}x")
