################################################################
# generate results for LoWo21 Figure 3
# r_min, fraction raindrop mass evaporated as functions 
# of vertical wind speed
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


# maximum stable raindrop size
# from Rayleigh-Taylor instability, ℓ_RT = 0.5πa
r_max = drop_prop.calc_r_max_RT(pl,np.pi/2.)
# set up grids
# space r0s closer for higher r0 values bc evaporation varies more
# space ws closer for higher w values bc evaporation varies more
n_r01 = 100
n_r02 = 200
n_r0 = n_r01 + n_r02
n_w1 = 30
n_w2 = 50
n_w = n_w1+n_w2
ws1 = np.linspace(-10,0,n_w1)
ws2 = np.linspace(0.1,10,n_w2)
ws = np.zeros(n_w) # [m/s]
ws[:n_w1] = ws1
ws[n_w1:] = ws2
r0s = np.zeros(n_r0) # [m]
r0s1 = np.logspace(np.log10(r_small)-1,np.log10(2e-4),n_r01)
r0s2 = np.logspace(np.log10(2.1e-4),np.log10(r_max),n_r02)
r0s[:n_r01]  = r0s1
r0s[n_r01:]  = r0s2
r0grid, wgrid  = np.meshgrid(r0s,ws)
m_frac_evap = np.zeros((n_w,n_r0)) # [ ]
r_mins = np.zeros(n_w) # [m]


for j,w in enumerate(ws):
    r_mins[j] = fall.calc_smallest_raindrop(pl,dr,w=w)[0] # [m]
    for i,r0 in enumerate(r0s):
        if fall.v_t_w_0(r0,w,pl)<0: # don't bother to integrate if |v_term(r0)| is smaller than w
            if r0>=r_mins[j]: # don't bother to integrate if r0 is smaller than rmin
                try:
                    sol = fall.integrate_fall(pl,r0,w=w)
                    if sol.status==0:
                        r_end = sol.sol.__call__(0.0)[0] # [m]
                    else: # sol.status!=0 means drop totally evaporated or evaporated small enough that it began moving upwards
                        r_end = dr
                except ValueError:
                    print('has ValueError','i',i,'j',j,'w',w,'r0',r0)
                    r_end = dr
            else:
                r_end = dr
        else:
            r_end = dr
        m_frac_evap[j,i] = 1 - r_end**3/r0**3 # [ ]

# save results
dir = 'output/fig03/'
np.save(dir+'r0grid', r0grid)
np.save(dir+'wgrid', wgrid)
np.save(dir+'m_frac_evap', m_frac_evap)
np.save(dir+'ws',ws)
np.save(dir+'r_mins',r_mins)
