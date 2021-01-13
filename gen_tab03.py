################################################################
# generate results for LoWo21 Table 3
# basic raindrop properties for common condensibles
# other than H2O
################################################################
import numpy as np
import pandas as pd

# properties of interest for common condensibles
c_props = np.genfromtxt('data/condensible_properties2.csv',delimiter=',',
                        skip_header=1,usecols=(1,2,3,4))
# names of each condensible species to write out
c_species = ['CH4','NH3','H2O','Fe','SiO2']

# following proportionality of LoWo21 eq (26)
# r_max ∝ √(σ/ρ_ℓ)
r_max_rats = (c_props[:,2]/c_props[:,1])**0.5
# evaporation energy for same radius raindrop
# E = m(r)*L ∝ L*ρ_ℓ
E_evap_rats = c_props[:,1]*c_props[:,3]
# set up list for outputting table results
tab03_list = []
# iterate over condensibles
for i in range(c_props.shape[0]):
    tab03_list.append([c_species[i],c_props[i,0],c_props[i,1],c_props[i,2],r_max_rats[i]/r_max_rats[2],c_props[i,3],E_evap_rats[i]/E_evap_rats[2]])
# convert to dataframe to easily save to csv file
tab03 = pd.DataFrame(tab03_list,columns=['condensible','T_melt [K]','rho_ell [kg/m^3]','sigma [N/m]','r_max [r_max,H2O]','L [J/kg]','E_evap(r) [E_evap(r,H2O)]'])
tab03.to_csv('tabs/tab03.csv',index=False)
