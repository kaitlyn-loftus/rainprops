import src.fall as fall
from src.planet import Planet
import src.drop_prop as drop_prop
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton,brentq
import time

dir = 'output/fig04/'

# panels (a) and (b)

X = np.zeros(5) # composition
X[2] = 1. # f_N2  [mol/mol]
T_surf = 300 # [K]
p_surf = 1.01325e5 # [Pa]
RH_surf = 0.75 # []
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
pl = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)

n_z = 100
zs = np.linspace(0,pl.z_LCL,n_z)
r_min = np.zeros((n_z-1,2))
t1 = time.time()
for i,z in enumerate(zs[:-1]):
    pl.z2x4drdz(pl.z_LCL)
    r_min[i,0] = fall.calc_smallest_raindrop(pl,z_end=z,dr=5e-7)[0]
t2 = time.time()
for i,z in enumerate(zs[:-1]):
    ell = pl.z_LCL-z
    r_min[i,1] = drop_prop.calc_r_from_Λ(pl,ell,Λ_val=1)
t3 = time.time()

print('t_int',t2-t1)
print('t_nond',t3-t2)


np.save(dir+'r_min',r_min)
np.save(dir+'zs',zs)

n_r = 200
dr = 1e-6
r0s = np.logspace(-5,-3,n_r)
ell = pl.z_LCL
m_frac_evap = np.zeros((n_r,2))
for i,r0 in enumerate(r0s):
    sol = fall.integrate_fall(pl,r0)
    if sol.status==0:
        r_end = sol.sol.__call__(0.0)[0]
    else:
        r_end = dr
    m_frac_evap[i,0] = 1 - r_end**3/r0**3
    m_frac_evap[i,1] = drop_prop.calc_Λ(r0,pl,ell)

m_frac_evap[np.where(m_frac_evap[:,1]>1.),1] = 1.
np.save(dir+'r0s',r0s)
np.save(dir+'m_frac_evap',m_frac_evap)

# panels (c) and (d)

X_H2 = np.zeros(5) # composition
X_N2 = np.zeros(5) # composition
X_CO2 = np.zeros(5) # composition
X_H2[0] = 1. # f_H2  [mol/mol]
X_N2[2] = 1. # f_N2 [mol/mol]
X_CO2[4] = 1. # f_CO2 [mol/mol]
T_LCL = 275 # [K]
p_LCL = 7.5e4 # [Pa]
RH = 1. # []
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
Xs = [X_H2,X_N2,X_CO2]

n_var = 10

p_LCLs = np.logspace(np.log10(5e3),7,n_var)
T_LCLs = np.linspace(275,400,n_var)
g = np.linspace(2,25,n_var)
var_char = np.array([p_LCLs,g,T_LCLs])

n_ℓ = 3
ℓs = np.array([100,500,1000])
r_mins = np.zeros((3,3,n_var,n_ℓ,2))
r0s = np.array([0.05,0.1,0.5,1])*1e-3
n_r0 = 4
m_frac_evap = np.zeros((3,3,n_var,n_r0,2))
dr = 5e-7 # [m]

for i,X in enumerate(Xs):
    for x in range(3):
        for j,v in enumerate(var_char[x,:]):
            if x==0:
                pl = Planet(R_p,T_LCL,v,X,'h2o',RH,M_p)
            elif x==1:
                pl = Planet(R_p,T_LCL,p_LCL,X,'h2o',RH,M_p,g_force=v)
            elif x==2:
                pl = Planet(R_p,v,p_LCL,X,'h2o',RH,M_p)
            for k,ℓ in enumerate(ℓs):
                print(i,x,j,k)
                r_mins[i,x,j,k,0] = fall.calc_smallest_raindrop(pl,z_end=-ℓ,dr=5e-7)[0]
                try:
                    r_mins[i,x,j,k,1] = drop_prop.calc_r_from_Λ(pl,ℓ)
                except ValueError:
                    r_mins[i,x,j,k,1] = None
                if k==1:
                    for q,r0 in enumerate(r0s):
                        sol = fall.integrate_fall(pl,r0,z_end=-ℓ)
                        if sol.status==0:
                            r_end = sol.sol.__call__(-ℓ)[0]
                        else:
                            r_end = 0.
                        m_frac_evap[i,x,j,q,0] = 1 - r_end**3/r0**3
                        Λ = drop_prop.calc_Λ(r0,pl,ℓ)
                        if Λ>1:
                            m_frac_evap[i,x,j,q,1] = 1.
                        else:
                            m_frac_evap[i,x,j,q,1] = Λ




percent_err_rmin = (r_mins[:,:,:,:,0] - r_mins[:,:,:,:,1])/r_mins[:,:,:,:,0]
rat_rmin = r_mins[:,:,:,:,1]/r_mins[:,:,:,:,0]
diff_rmin = r_mins[:,:,:,:,1]/r_mins[:,:,:,:,0]

percent_err_fevap = (m_frac_evap[:,:,:,:,0] - m_frac_evap[:,:,:,:,1])/m_frac_evap[:,:,:,:,0]
rat_fevap = m_frac_evap[:,:,:,:,1]/m_frac_evap[:,:,:,:,0]
diff_fevap = m_frac_evap[:,:,:,:,1]-m_frac_evap[:,:,:,:,0]

np.save(dir+'var_char',var_char)
np.save(dir+'ells_broad',ℓs)
np.save(dir+'r0s_broad',r0s)
np.save(dir+'r_mins_broad',r_mins)
np.save(dir+'percent_err_rmin_broad',percent_err_rmin)
np.save(dir+'rat_rmin_broad',rat_rmin)
np.save(dir+'diff_rmin_broad',diff_rmin)

np.save(dir+'percent_err_fevap_broad',percent_err_fevap)
np.save(dir+'rat_fevap_broad',rat_fevap)
np.save(dir+'diff_fevap_broad',diff_fevap)
np.save(dir+'m_frac_evap_broad',m_frac_evap)
