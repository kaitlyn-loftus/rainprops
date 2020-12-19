import numpy as np
from src.planet import Planet
import src.fall as fall
import src.drop_prop as drop_prop
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

dir = 'output/fig07/'

# set up planets differing only by dry gas composition
X_h2 = np.zeros(5) # composition
X_n2 = np.zeros(5) # composition
X_co2 = np.zeros(5) # composition
X_h2[0] = 1. # pure H2 atm
X_n2[2] = 1. # pure N2 atm
X_co2[4] = 1. # pure CO2 atm
T_LCL = 275 # [K]
p_LCL = 7.5e4 # [Pa]
RH_LCL = 1. # []
R_p = 1 # [R_earth]
M_p = 1 # [M_earth]
pl_h2 = Planet(R_p,T_LCL,p_LCL,X_h2,'h2o',RH_LCL,M_p)
pl_n2 = Planet(R_p,T_LCL,p_LCL,X_n2,'h2o',RH_LCL,M_p)
pl_co2 = Planet(R_p,T_LCL,p_LCL,X_co2,'h2o',RH_LCL,M_p)

pl_labels = ['h2','n2','co2']
planets = [pl_h2,pl_n2,pl_co2]

# some representative initial radii
r0s = np.array([0.1e-3,0.5e-3,1e-3])*2 # [m]
n_r0 = 3
n_z1 = 1500
n_z2 = 6000
n_z = n_z1 + n_z2
n_pl = len(planets)
zs = np.zeros(n_z)


for i,r0 in enumerate(r0s):
    for j,pl in enumerate(planets):
        # integrate falling of raindrop of given size r0 for given planet pl
        sol = fall.integrate_fall_w_t(pl,r0)
        # z where evaporate to r = 1e-6
        z_end = fall.calc_last_z(sol) # [m]
        # space zs with extra resolution at the end of raindrop's evaporation
        zs[:n_z1] = np.linspace(pl.z_LCL,z_end +(pl.z_LCL-z_end)*0.04,n_z1)
        zs[n_z1:] = np.linspace(z_end+(pl.z_LCL-z_end)*0.04-0.001,z_end,n_z2)
        ts = sol.sol.__call__(zs)[0,:] # [s]
        t2z = interp1d(ts, zs) # interpolate to get a function to convert t values to z values
        t_end = ts[-1] # [s] last t
        ts_evap0 = np.arange(0,t_end,1.) # t in seconds
        ts_evap = np.zeros(ts_evap0.shape[0]+1)
        ts_evap[:-1] = ts_evap0
        ts_evap[-1] = t_end # + difference between full second and last time step
        zs_evap = t2z(ts_evap) # convert t to z
        ps_evap = pl.z2p(zs_evap) # convert z to p from HSE
        rs_evap = sol.sol.__call__(zs_evap)[1,:] # calculate r from z from integration
        m_evapdt = 4./3.*np.pi*pl.c.ρ_const*(rs_evap[:-1]**3-rs_evap[1:]**3)*r0s[2]**3/r0s[i]**3 # [kg/s] mass evaporated per second
        P_evap = m_evapdt*pl.c.L(pl.z2T(zs_evap[1:])) # [W] power evaporated
        P_evap[-1] = P_evap[-1]*(ts_evap[-1] - ts_evap[-2]) # adjust last entry because not a full second
        # save outputs by planet and r0 because spacing differs
        np.save(dir+'power_evap_norm'+'_'+pl_labels[j]+'_'+str(i),P_evap)
        np.save(dir+'p_evap_norm'+'_'+pl_labels[j]+'_'+str(i),ps_evap)
        np.save(dir+'ts_evap'+'_'+pl_labels[j]+'_'+str(i),ts_evap)
        np.save(dir+'zs_evap'+'_'+pl_labels[j]+'_'+str(i),zs_evap)
# save r0s
np.save(dir+'r0s',r0s)
