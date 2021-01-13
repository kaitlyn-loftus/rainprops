################################################################
# generate results for LoWo21 Figure 6
# raindrop size bounds across different planetary conditions
################################################################
import numpy as np
from src.planet import Planet
import src.fall as fall
import src.drop_prop as drop_prop

# set up default planet characteristics
# default state is Earth-like
# vary g, p_surf, T in isolation
R_p = 1. # [Earth radii]
M_p = 1. # [Earth masses]
T_surf = 300. # [K]
p_surf = 1.01325e5 # [Pa]
RH_surf = 0.5 # [ ]
f_h2 = 0.  # [mol/mol]
f_he = 0. # [mol/mol]
f_n2 = 1. # [mol/mol]
f_o2 = 0. # [mol/mol]
f_co2 = 0. # [mol/mol]
# H2, He, N2, O2, CO2
atm_comp = np.array([f_h2,f_he,f_n2,f_o2,f_co2]) # [mol/mol] dry volume mixing concentrations

# number of points on line
n = 25

# varying characteristics
p_surfs = np.logspace(np.log10(5e3),7,n) # [Pa]
T_surfs = np.linspace(300,450,n) # [K]
g = np.linspace(2,25,n) # [m/s2]
var_char = np.array([p_surfs,g,T_surfs]) # array of varying planetary conditions

# arrays to store max and min bounds
r_maxs = np.zeros((3,n)) # [m]
r_mins = np.zeros((3,n,3)) # [m]


RHs = [0.25,0.5,0.75] # [ ] set 3 RH values to show effect on r_min

for i in range(3):
    for j,v in enumerate(var_char[i,:]):
        for k,RH in enumerate(RHs):
            # set up planetary conditions
            if i==0:
                pl = Planet(R_p,T_surf,v,atm_comp,'h2o',RH,M_p)
            elif i==1:
                pl = Planet(R_p,T_surf,p_surf,atm_comp,'h2o',RH,M_p,g_force=v)
            elif i==2:
                pl = Planet(R_p,v,p_surf,atm_comp,'h2o',RH,M_p)
            # calculate r_min to reach surface under given planetary conditions
            r_mins[i,j,k] = fall.calc_smallest_raindrop(pl,1e-6)[0]

            if k==1:
                # reset conditions to surface
                pl.z2x4drdz(0.)
                # maximum stable raindrop size
                # from Rayleigh-Taylor instability, ℓ_RT = 0.5πa
                r_maxs[i,j] = drop_prop.calc_r_max_RT(pl,np.pi/2.)


# save results
dir = 'output/fig06/'
np.save(dir+'r_maxs',r_maxs)
np.save(dir+'r_mins',r_mins)
np.save(dir+'var_char',var_char)
np.save(dir+'RHs',np.array(RHs))
