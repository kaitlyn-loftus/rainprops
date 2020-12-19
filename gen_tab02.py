import numpy as np
from src.planet import Planet
import src.fall as fall
import src.drop_prop as drop_prop
import pandas as pd

# set up different atm composition planets
# X_species array holds dry atmospheric molar/volume concentrations
# for each planet
X_H2 = np.zeros(5)
X_He = np.zeros(5)
X_N2 = np.zeros(5)
X_O2 = np.zeros(5)
X_CO2 = np.zeros(5)
X_H2[0] = 1. # f_H2 [mol/mol]
X_He[1] = 1. # f_He [mol/mol]
X_N2[2] = 1. # f_N2 [mol/mol]
X_O2[3] = 1. # f_O2 [mol/mol]
X_CO2[4] = 1. # f_CO2 [mol/mol]
T_LCL = 275 # [K]
p_LCL = 7.5e4 # [Pa]
RH_LCL = 1. # []
# Earth gravity
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
pl_H2 = Planet(R_p,T_LCL,p_LCL,X_H2,'h2o',RH_LCL,M_p)
pl_He = Planet(R_p,T_LCL,p_LCL,X_He,'h2o',RH_LCL,M_p)
pl_N2 = Planet(R_p,T_LCL,p_LCL,X_N2,'h2o',RH_LCL,M_p)
pl_O2 = Planet(R_p,T_LCL,p_LCL,X_O2,'h2o',RH_LCL,M_p)
pl_CO2 = Planet(R_p,T_LCL,p_LCL,X_CO2,'h2o',RH_LCL,M_p)
pls = [pl_H2,pl_He,pl_N2,pl_O2,pl_CO2]

r0 = 5e-4 # [m] initial raindrop radius at LCL
n_N2 = 2 # index for N2 atm
atm_label = ['H2','He','N2','O2','CO2']
effects_label = ['all','H','v','transport']
tab02_list = [] # set up list for outputting table results

# iterate over different composition planets
for i,pl in enumerate(pls):
    # all effects of composition together
    sol = fall.integrate_fall_w_t(pl,r0)
    z_evap = fall.calc_last_z(sol) # [m]
    t_evap = sol.sol.__call__(z_evap)[0] # [s]
    f_H_evap = -z_evap/pl.calc_H(0.) # [m/m]
    tab02_list.append([atm_label[i],effects_label[0],t_evap,-z_evap,f_H_evap])
    if i!=n_N2: # don't do individual composition effects for N2 (use N2 as neutral background)
        # only atmospheric structure
        sol = fall.integrate_fall_w_t_w_3pl(pl,pl_N2,pl_N2,r0)
        z_evap = fall.calc_last_z(sol) # [m]
        t_evap = sol.sol.__call__(z_evap)[0] # [s]
        f_H_evap = -z_evap/pl.calc_H(0.) # [m/m]
        tab02_list.append([atm_label[i],effects_label[1],t_evap,-z_evap,f_H_evap])
        # only velocity of drop
        sol = fall.integrate_fall_w_t_w_3pl(pl_N2,pl,pl_N2,r0)
        z_evap = fall.calc_last_z(sol) # [m]
        t_evap = sol.sol.__call__(z_evap)[0] # [s]
        f_H_evap = -z_evap/pl.calc_H(0.) # [m/m]
        tab02_list.append([atm_label[i],effects_label[2],t_evap,-z_evap,f_H_evap])
        # only condensible mass/heat transport
        sol = fall.integrate_fall_w_t_w_3pl(pl_N2,pl_N2,pl,r0)
        z_evap = fall.calc_last_z(sol) # [m]
        t_evap = sol.sol.__call__(z_evap)[0] # [s]
        f_H_evap = -z_evap/pl.calc_H(0.) #[m/m]
        tab02_list.append([atm_label[i],effects_label[3],t_evap,-z_evap,f_H_evap])
# convert to dataframe to easily save to csv file
tab02 = pd.DataFrame(tab02_list,columns=['atm gas','effect','t_evap [s]','z_evap [m]','z_evap/H []'])
tab02.to_csv('tabs/tab02.csv',index=False)
