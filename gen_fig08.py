import src.fall as fall
from src.planet import Planet
import src.drop_prop as drop_prop
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton,brentq
import time


# set up different planets

# EARTH
X = np.zeros(5) # composition
X[2] = 0.8 # f_N2  [mol/mol]
X[3] = 0.2 # f_O2  [mol/mol]
T_surf = 290. # [K]
p_surf = 1.01325e5 # [Pa]
RH_surf = 0.75 # []
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
Earth = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)

# MARS
X = np.zeros(5) # composition
X[4] = 1. # f_CO2  [mol/mol]
T_surf = 290. # [K] ???
p_surf = 2e5 # [Pa]
RH_surf = 0.75 # [] ????
R_p = 0.532 # [R_earth]
M_p = 0.107 # [M_earth]
Mars = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)
print('Mars g',Mars.g)

# gas giant compositions from Leconte et al. (2017)
# and cloud levels from Carlson et al. (1988)

# JUPITER
X = np.zeros(5) # composition
X[1] =  0.136 # f_He  [mol/mol]
X[0] = 1 -  X[1] # f_H2  [mol/mol]
T_LCL = 274. # [K]
p_LCL = 4.85e5 # [Pa]
RH_LCL = 1. # []
R_p = 11.2089 # [R_earth]
M_p = 317.8 # [M_earth]
Jupiter = Planet(R_p,T_LCL,p_LCL,X,'h2o',RH_LCL,M_p)
print('Jupiter g',Jupiter.g)

# SATURN
X = np.zeros(5) # composition
X[1] =  0.12 # f_He  [mol/mol]
X[0] = 1 -  X[1] # f_H2  [mol/mol]
T_LCL = 284. # [K]
p_LCL = 10.4e5 # [Pa]
RH_LCL = 1. # []
R_p = 9.449 # [R_earth]
M_p = 95.16 # [M_earth]
Saturn = Planet(R_p,T_LCL,p_LCL,X,'h2o',RH_LCL,M_p)
print('Saturn g',Saturn.g)

# K2-18b
X = np.zeros(5) # composition
X[1] =  0.10 # f_He  [mol/mol] ???
X[0] = 1 -  X[1] # f_H2  [mol/mol]
T_LCL = 275 # [K] ???
p_LCL = 1e4 # [Pa] ???
RH_LCL = 1. # []
R_p = 2.610 # [R_earth] Benneke et al. 2019
M_p = 8.63 # [M_earth] Cloutier et al. 2019 +- 1.35 M_earth
K2_18b = Planet(R_p,T_LCL,p_LCL,X,'h2o',RH_LCL,M_p)
print('K2-18b g',K2_18b.g)


planets = [Earth,Mars,Jupiter,Saturn,K2_18b]
planet_names = ['Earth','Mars','Jupiter','Saturn','K2-18b']

n_z = 25
H_LCL = np.zeros(len(planets))
r_min = np.zeros((n_z,len(planets),2))
r_maxs = np.zeros(len(planets))
z_maxs = np.zeros(len(planets))
zH_maxs = np.zeros(len(planets))
zs = np.zeros((n_z,len(planets)))
zHs = np.zeros((n_z,len(planets)))
for j,pl in enumerate(planets):
    # maximum stable raindrop size
    # from Rayleigh-Taylor instability, ℓ_RT = 0.5πa
    r_maxs[j] = drop_prop.calc_r_max_RT(pl,np.pi/2.)
    # calculate scale height at LCL to normalize against
    H_LCL[j] = pl.calc_H(pl.z_LCL)

    # set max Δz to consider
    # distinguish between planets with surface & no surface
    # for planets with no surface, end z set from where r_max
    # finishes evaporating
    if pl.z_LCL!=0:
        z_maxs[j] = pl.z_LCL
    else:
        sol_rmax = fall.integrate_fall(pl,r_maxs[j])
        z_maxs[j] = -fall.calc_last_z(sol_rmax)
    # equally space z in meters in logspace
    zs[:,j] = np.logspace(1,np.log10(z_maxs[j])-0.01,n_z)
    # equally space z in H in logspace
    zH_maxs[j] = z_maxs[j]/H_LCL[j]
    zHs[:,j] = np.logspace(-3,np.log10(z_maxs[j]/H_LCL[j])-0.01,n_z)


# go through planets
for j,pl in enumerate(planets):
    # go through each distance [m]
    for i,z in enumerate(zs[:,j]):
        # reset planet z to cloud base
        pl.z2x4drdz(pl.z_LCL)
        # calc end z from where cloud base is
        z_end = pl.z_LCL - z
        if z_maxs[j]<z:
            r_min[i,j,0] = None
        else:
            # calc r_min for that z
            r_min[i,j,0] = fall.calc_smallest_raindrop(pl,z_end=z_end,dr=5e-7)[0]
    # go through each altitude [H] & do same thing
    for i,zH in enumerate(zHs[:,j]):
        pl.z2x4drdz(pl.z_LCL)
        z = zH*H_LCL[j]
        z_end = pl.z_LCL - z
        if z_maxs[j]<z:
            r_min[i,j,1] = None
        else:
            r_min[i,j,1] = fall.calc_smallest_raindrop(pl,z_end=z_end,dr=5e-7)[0]

# output results
dir = 'output/fig08/'
np.save(dir+'r_mins_log',r_min)
np.save(dir+'zs_log',zs)
np.save(dir+'zHs_log',zHs)
np.save(dir+'r_maxs',r_maxs)
np.save(dir+'z_maxs',z_maxs)
np.save(dir+'zH_maxs',zH_maxs)
