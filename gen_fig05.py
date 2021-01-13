################################################################
# generate results for LoWo21 Figure 5
# pressure dependence of r_max using different calculation methods
################################################################
import numpy as np
from src.planet import Planet
import src.fall as fall
import src.drop_prop as drop_prop

# set up baseline planet characteristics from Earth-like conditions
R_p = 1. # [Earth radii]
M_p = 1. # [Earth masses]
T_surf = 275. # [K]
RH_surf = 0. # [ ]
f_h2 = 0.  # [mol/mol]
f_he = 0. # [mol/mol]
f_n2 = 1. # [mol/mol]
f_o2 = 0. # [mol/mol]
f_co2 = 0. # [mol/mol]
atm_comp = np.array([f_h2,f_he,f_n2,f_o2,f_co2]) # [mol/mol] dry volume mixing concentrations

# number of points on line
n = 100

# varying planetary characteristics
ps = np.logspace(3,7,n) # [Pa]
ρs = np.zeros(n) # [kg/m3], linearly will map to p as T constant and RH=0
r_maxs = np.zeros((n,9)) # [m]

RH = 0 # [ ] set RH to 0 so p maps linearly to ρ

for i,p in enumerate(ps):
    # for each pressure p calculate all different r_max methods
    # set up planetary conditions
    pl = Planet(R_p,T_surf,p,atm_comp,'h2o',RH,M_p,is_find_LCL=False)
    # record air density
    ρs[i] = pl.ρ # [kg/m3]

    # Craddock & Lorenz (2017)
    r_maxs[i,0] = drop_prop.calc_r_max_CL17(pl)

    # Lorenz (1993)
    r_maxs[i,1] = drop_prop.calc_r_max_L93(pl)

    # Palumbo et al. (2020)
    r_maxs[i,2] = drop_prop.calc_r_max_P20(pl)

    # force balance, ℓ_σ = 2πreq
    r_maxs[i,3] = drop_prop.calc_r_max_fb_req(pl)

    # force balance, ℓ_σ = 2πa
    r_maxs[i,4] = drop_prop.calc_r_max_fb_a(pl)

    # Rayleigh-Taylor instability, ℓ_RT = 0.5πa
    r_maxs[i,5] = drop_prop.calc_r_max_RT(pl,np.pi/2.)

    # Rayleigh-Taylor instability, ℓ_RT = 2a
    r_maxs[i,6] = drop_prop.calc_r_max_RT(pl,2.)

    # Rayleigh-Taylor instability, ℓ_RT = 0.5πreq
    r_maxs[i,7] = drop_prop.calc_r_max_RT_r(pl,np.pi/2)

    # Rayleigh-Taylor instability, ℓ_RT = 2req
    r_maxs[i,8] = drop_prop.calc_r_max_RT_r(pl,2.)

# save outputs
dir = 'output/fig05/'
np.save(dir+'r_maxs',r_maxs)
np.save(dir+'ps',ps)
np.save(dir+'rhos',ρs)
