from src.planet import Planet
import src.fall as fall
import numpy as np
import src.drop_prop as drop_prop

n_r0 = 6
n_z = 1500


X = np.zeros(5) # composition
X[2] = 1. # f_N2  [mol/mol]
T_surf = 300 # [K]
p_surf = 1.01325e5 # [Pa]
RH_surf = 0.75 # []
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
pl = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)

dr = 1e-6 # [m]
r_small, int_descrip = fall.calc_smallest_raindrop(pl,dr,is_integrate_Tdrop=True)

r0s = np.logspace(np.log10(r_small/2.),np.log10(r_small*2),n_r0)
zs = np.zeros((n_r0,n_z))
rs = np.zeros((n_r0,n_z))
ts = np.zeros((n_r0,n_z))

pl.z2x4drdz(0)

for i,r0 in enumerate(r0s):
    sol = fall.integrate_fall_w_t(pl,r0)
    z_end = fall.calc_last_z(sol)
    zs[i,:] = np.linspace(pl.z_LCL,z_end,n_z)
    rs[i,:] = sol.sol.__call__(zs[i,:])[1,:]
    ts[i,:] = sol.sol.__call__(zs[i,:])[0,:]

dir = 'output/fig01/'
np.save(dir+'zs',zs)
np.save(dir+'rs',rs)
np.save(dir+'ts',ts)
np.save(dir+'r0s',r0s)
