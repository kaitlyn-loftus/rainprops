import numpy as np
from src.planet import Planet
import src.fall as fall
import src.drop_prop as drop_prop
import matplotlib.pyplot as plt

dir = 'output/sfig01/'

X_H2 = np.zeros(5) # composition
X_N2 = np.zeros(5) # composition
X_CO2 = np.zeros(5) # composition
X_H2[0] = 1. # f_H2  [mol/mol]
X_N2[2] = 1. # f_N2 [mol/mol]
X_CO2[4] = 1. # f_CO2 [mol/mol]
T_LCL = 280 # [K]
p_LCL = 1e4 # [Pa]
RH = 1. # []
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
Xs = [X_H2,X_N2,X_CO2]

n = 10

p_LCLs = np.logspace(np.log10(5e3),7,n)
T_LCLs = np.linspace(275,400,n)
g = np.linspace(2,25,n)
var_char = np.array([p_LCLs,g,T_LCLs])

z_evap = np.zeros((3,3,n))
t_evap = np.zeros((5,3,n))
f_H_evap = np.zeros((5,3,n,2))

f_vterm = 0.99
t_vterm = np.zeros((3,3,n))
t_evap = np.zeros((3,3,n))
z_vterm = np.zeros((3,3,n))
z_H_vterm = np.zeros((3,3,n))
r_maxs = np.zeros((3,3,n))
t_t = np.zeros((3,3,n))
z_z = np.zeros((3,3,n))

n_t = 1000
r0s = np.array([0.05,0.1,0.5,1])*1e-3
v_error = np.zeros((3,3,n,3,4))
# max median mean
vs = np.zeros((3,3,n,n_t,2,4))



for i,X in enumerate(Xs):
    for x in range(3):
        for j,v in enumerate(var_char[x,:]):
            if x==0:
                pl = Planet(R_p,T_LCL,v,X,'h2o',RH,M_p)
            elif x==1:
                pl = Planet(R_p,T_LCL,p_LCL,X,'h2o',RH,M_p,g_force=v)
            elif x==2:
                pl = Planet(R_p,v,p_LCL,X,'h2o',RH,M_p)

            # maximum stable raindrop size
            # from Rayleigh-Taylor instability, ℓ_RT = 0.5πa
            r_maxs[i,x,j] = drop_prop.calc_r_max_RT(pl,np.pi/2.)
            sol = fall.integrate_2_vterm_w_a(pl,r_maxs[i,x,j],f_vterm=f_vterm)
            t_vterm[i,x,j] = fall.calc_t2vterm(sol,pl)


            z_vterm[i,x,j] = -sol.sol.__call__(t_vterm[i,x,j])[1]
            z_H_vterm[i,x,j] = z_vterm[i,x,j]/pl.calc_H(0.)

            sol = fall.integrate_fall_w_a(pl,r_maxs[i,x,j],f_v0_v_term=0.)
            t_evap[i,x,j] = fall.calc_last_t(sol)
            z_evap[i,x,j] = -sol.sol.__call__(t_evap[i,x,j])[1]
            t_t[i,x,j] = t_vterm[i,x,j]/t_evap[i,x,j]
            z_z[i,x,j] = z_vterm[i,x,j]/z_evap[i,x,j]

            for q,r0 in enumerate(r0s):
                pl.z2x4drdz(0.)
                sol = fall.integrate_fall_w_a(pl,r0,f_v0_v_term=1.)
                t_end_a = fall.calc_last_t(sol)
                ts = np.linspace(0,t_end_a,n_t)
                output_w_a = sol.sol.__call__(ts)
                vs[i,x,j,:,0,q] = output_w_a[2,:]
                for k in range(n_t):
                    pl.z2x4drdz(output_w_a[1,k])
                    vs[i,x,j,k,1,q] = -drop_prop.calc_v(output_w_a[0,k],pl,pl.f_rat,pl.f_C_D)


                rel_err = np.abs((vs[i,x,j,:,1,q] - vs[i,x,j,:,0,q])/vs[i,x,j,:,0,q])
                v_error[i,x,j,0,q] = np.amax(rel_err)
                v_error[i,x,j,1,q] = np.median(rel_err)
                v_error[i,x,j,2,q] = np.mean(rel_err)
                diff_err = np.abs((vs[i,x,j,:,1,q] - vs[i,x,j,:,0,q]))


np.save(dir+'r_maxs',r_maxs)
np.save(dir+'t_vterms',t_vterm)
np.save(dir+'z_vterms',z_vterm)
np.save(dir+'z_H_vterm',z_H_vterm)
np.save(dir+'v_error',v_error)
np.save(dir+'t_vterm_t_evap',t_t)
np.save(dir+'z_vterm_z_evap',z_z)
np.save(dir+'vs',vs)
