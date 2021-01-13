################################################################
# generate results for LoWo21 Figure 1
# evaporating raindrops r(z) for different r0s near r_min
################################################################
from src.planet import Planet
import src.fall as fall
import numpy as np
import src.drop_prop as drop_prop

# number of r0 and z points
n_r0 = 6
n_z = 1500

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
r_small, int_descrip = fall.calc_smallest_raindrop(pl,dr,is_integrate_Tdrop=True)

# make r0s evenly logspaced around r_min for these planetary conditions
r0s = np.logspace(np.log10(r_small/2.),np.log10(r_small*2),n_r0)
zs = np.zeros((n_r0,n_z))
rs = np.zeros((n_r0,n_z))
ts = np.zeros((n_r0,n_z))

# integrate each r0 from LCL until evaporation or hitting the surface
for i,r0 in enumerate(r0s):
    sol = fall.integrate_fall_w_t(pl,r0) # integrate
    z_end = fall.calc_last_z(sol) # [m]
    zs[i,:] = np.linspace(pl.z_LCL,z_end,n_z) # [m]
    rs[i,:] = sol.sol.__call__(zs[i,:])[1,:] # [m]
    ts[i,:] = sol.sol.__call__(zs[i,:])[0,:] # [s]

# save results
dir = 'output/fig01/'
np.save(dir+'zs',zs)
np.save(dir+'rs',rs)
np.save(dir+'ts',ts)
np.save(dir+'r0s',r0s)
