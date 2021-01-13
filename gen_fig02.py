################################################################
# generate results for LoWo21 Figure 2
# r_min, fraction raindrop mass evaporated as functions of RH
################################################################
import numpy as np
from src.planet import Planet
import src.fall as fall
import src.drop_prop as drop_prop

# set up planetary conditions
X = np.zeros(5) # composition
X[2] = 1. # f_N2  [mol/mol]
T_surf = 300 # [K]
p_surf = 1.01325e5 # [Pa]
RH_surf = 0.75 # [ ]
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
pl = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)

dr = 1e-6 # [m]
r_small, int_descrip = fall.calc_smallest_raindrop(pl,dr)
n_r0 = 150
n_RH1 = 30
n_RH2 = 30
n_RH = n_RH1 + n_RH2
# maximum stable raindrop size
# from Rayleigh-Taylor instability, ℓ_RT = 0.5πa
r_max = drop_prop.calc_r_max_RT(pl,np.pi/2.) # [m]
r0s = np.logspace(np.log10(r_small)-1,np.log10(r_max),n_r0) # [m]
RHs1 = np.linspace(0.25,0.75,n_RH1) # [ ] space RHs closer for lower RH values bc evaporation varies more
RHs2 = np.linspace(0.76,0.99,n_RH2) # [ ]
RHs = np.zeros(n_RH)
RHs[:n_RH1] = RHs1
RHs[n_RH1:] = RHs2
r0grid, RHgrid  = np.meshgrid(r0s,RHs)
m_frac_evap = np.zeros((n_RH,n_r0)) # [ ]
r_mins = np.zeros(n_RH) # [m]
for j,RH in enumerate(RHs):
    pl = Planet(R_p,T_surf,p_surf,X,'h2o',RH,M_p)
    r_mins[j] = fall.calc_smallest_raindrop(pl,dr)[0] # [m]
    for i,r0 in enumerate(r0s):
        if r0>=r_mins[j]: # don't bother to integrate if r0 is smaller than rmin
            sol = fall.integrate_fall(pl,r0)
            z_end = fall.calc_last_z(sol) # [m]
            r_end = sol.sol.__call__(z_end)[0] # [m]
            m_frac_evap[j,i] = 1 - r_end**3/r0**3 # [ ]
        else:
            m_frac_evap[j,i] = 1. # [ ]

# save results
dir = 'output/fig02/'
np.save(dir+'r0grid', r0grid)
np.save(dir+'RHgrid', RHgrid)
np.save(dir+'m_frac_evap', m_frac_evap)
np.save(dir+'RHs',RHs)
np.save(dir+'r_mins',r_mins)
