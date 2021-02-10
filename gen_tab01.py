################################################################
# generate results for LoWo21 Table 1
# planetary properties used in calculations
# all planetary objects generated in one place for additional
# information beyond what's given in Table 1
################################################################
import numpy as np
from src.planet import Planet
import src.fall as fall
import pandas as pd

tab01_list = [] # set up list for outputting table results

# set up different planets
# ??? indicates very guessed from a large range of possible values
# for specific planet cases

# EARTH-LIKE
X = np.zeros(5) # composition
X[2] = 1. # f_N2  [mol/mol]
T_surf = 300 # [K]
p_surf = 1.01325e5 # [Pa]
RH_surf = 0.75 # [ ]
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
Earth_like = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)
tab01_list.append(['Earth-like','surface',T_surf,'%.5E'%p_surf,RH_surf,'%.3E'%Earth_like.g,'%.3E'%Earth_like.calc_H(Earth_like.z_LCL),0,0,1,0,0])

# EARTH
X = np.zeros(5) # composition
X[2] = 0.8 # f_N2  [mol/mol]
X[3] = 0.2 # f_O2  [mol/mol]
T_surf = 290. # [K]
p_surf = 1.01325e5 # [Pa]
RH_surf = 0.75 # [ ]
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
Earth = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)
tab01_list.append(['Earth','surface',T_surf,'%.5E'%p_surf,RH_surf,'%.3E'%Earth.g,'%.3E'%Earth.calc_H(Earth.z_LCL),0,0,0.8,0.2,0])

# EARLY MARS
X = np.zeros(5) # composition
X[4] = 1. # f_CO2  [mol/mol]
T_surf = 290. # [K] ???
p_surf = 2e5 # [Pa]
RH_surf = 0.75 # [ ] ???
R_p = 0.532 # [R_earth]
M_p = 0.107 # [M_earth]
Mars = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)
tab01_list.append(['early Mars','surface',T_surf,'%.3E'%p_surf,RH_surf,'%.3E'%Mars.g,'%.3E'%Mars.calc_H(Mars.z_LCL),0,0,0,0,1])

# gas giant compositions from Leconte et al. (2017)
# and cloud levels from Carlson et al. (1988)
# note g for gas giants does NOT account for
# centrifugal force or oblateness

# JUPITER
X = np.zeros(5) # composition
X[1] =  0.136 # f_He  [mol/mol]
X[0] = 1 -  X[1] # f_H2  [mol/mol]
T_LCL = 274. # [K]
p_LCL = 4.85e5 # [Pa]
RH_LCL = 1. # [ ]
R_p = 11.2089 # [R_earth]
M_p = 317.8 # [M_earth]
Jupiter = Planet(R_p,T_LCL,p_LCL,X,'h2o',RH_LCL,M_p)
tab01_list.append(['Jupiter','LCL',T_LCL,'%.3E'%p_LCL,RH_LCL,'%.3E'%Jupiter.g,'%.3E'%Jupiter.calc_H(Jupiter.z_LCL),X[0],X[1],0,0,0])

# SATURN
X = np.zeros(5) # composition
X[1] =  0.12 # f_He  [mol/mol]
X[0] = 1 -  X[1] # f_H2  [mol/mol]
T_LCL = 284. # [K]
p_LCL = 10.4e5 # [Pa]
RH_LCL = 1. # [ ]
R_p = 9.449 # [R_earth]
M_p = 95.16 # [M_earth]
Saturn = Planet(R_p,T_LCL,p_LCL,X,'h2o',RH_LCL,M_p)
tab01_list.append(['Saturn','LCL',T_LCL,'%.3E'%p_LCL,RH_LCL,'%.3E'%Saturn.g,'%.3E'%Saturn.calc_H(Saturn.z_LCL),X[0],X[1],0,0,0])
# K2-18b
X = np.zeros(5) # composition
X[1] =  0.10 # f_He  [mol/mol] ???
X[0] = 1 -  X[1] # f_H2  [mol/mol]
T_LCL = 275 # [K] ???
p_LCL = 1e4 # [Pa] ???
RH_LCL = 1. # [ ]
R_p = 2.610 # [R_earth] Benneke et al. (2019)
M_p = 8.63 # [M_earth] Cloutier et al. (2019) +- 1.35 M_earth
K2_18b = Planet(R_p,T_LCL,p_LCL,X,'h2o',RH_LCL,M_p)
tab01_list.append(['K2-18b','LCL',T_LCL,'%.3E'%p_LCL,RH_LCL,'%.3E'%K2_18b.g,'%.3E'%K2_18b.calc_H(K2_18b.z_LCL),X[0],X[1],0,0,0])

# COMPOSITION
# set up planets differing only by dry gas composition
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
RH_LCL = 1. # [ ]
# Earth gravity
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
pl_H2 = Planet(R_p,T_LCL,p_LCL,X_H2,'h2o',RH_LCL,M_p)
pl_He = Planet(R_p,T_LCL,p_LCL,X_He,'h2o',RH_LCL,M_p)
pl_N2 = Planet(R_p,T_LCL,p_LCL,X_N2,'h2o',RH_LCL,M_p)
pl_O2 = Planet(R_p,T_LCL,p_LCL,X_O2,'h2o',RH_LCL,M_p)
pl_CO2 = Planet(R_p,T_LCL,p_LCL,X_CO2,'h2o',RH_LCL,M_p)
tab01_list.append(['composition-h2','LCL',T_LCL,'%.3E'%p_LCL,RH_LCL,'%.3E'%pl_H2.g,'%.3E'%pl_H2.calc_H(pl_H2.z_LCL),1,0,0,0,0])
tab01_list.append(['composition-he','LCL',T_LCL,'%.3E'%p_LCL,RH_LCL,'%.3E'%pl_He.g,'%.3E'%pl_He.calc_H(pl_He.z_LCL),0,1,0,0,0])
tab01_list.append(['composition-n2','LCL',T_LCL,'%.3E'%p_LCL,RH_LCL,'%.3E'%pl_N2.g,'%.3E'%pl_N2.calc_H(pl_N2.z_LCL),0,0,1,0,0])
tab01_list.append(['composition-o2','LCL',T_LCL,'%.3E'%p_LCL,RH_LCL,'%.3E'%pl_O2.g,'%.3E'%pl_O2.calc_H(pl_O2.z_LCL),0,0,0,1,0])
tab01_list.append(['composition-co2','LCL',T_LCL,'%.3E'%p_LCL,RH_LCL,'%.3E'%pl_CO2.g,'%.3E'%pl_CO2.calc_H(pl_CO2.z_LCL),0,0,0,0,1])

# BROAD
# n_Xs*n_var*n_var_char = 90 different planetary conditions
X_H2 = np.zeros(5) # composition
X_N2 = np.zeros(5) # composition
X_CO2 = np.zeros(5) # composition
X_H2[0] = 1. # f_H2  [mol/mol]
X_N2[2] = 1. # f_N2 [mol/mol]
X_CO2[4] = 1. # f_CO2 [mol/mol]
T_LCL = 275 # [K]
p_LCL = 7.5e4 # [Pa]
RH = 1. # [ ]
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
Xs = [X_H2,X_N2,X_CO2] # array of varying atm compositions
n_var = 10
p_LCLs = np.logspace(np.log10(5e3),7,n_var) # [Pa]
T_LCLs = np.linspace(275,400,n_var) # [K]
g = np.linspace(2,25,n_var) # [m/s2]
var_char = np.array([p_LCLs,g,T_LCLs]) # array of varying planetary characteristics
H_LCLs = np.zeros((3,3,n_var))
for i,X in enumerate(Xs): # iterate over dry gas composition
    for x in range(3): # iterate over p_LCLs, g, T_LCLs
        for j,v in enumerate(var_char[x,:]):
            if x==0:
                pl = Planet(R_p,T_LCL,v,X,'h2o',RH,M_p)
            elif x==1:
                pl = Planet(R_p,T_LCL,p_LCL,X,'h2o',RH,M_p,g_force=v)
            elif x==2:
                pl = Planet(R_p,v,p_LCL,X,'h2o',RH,M_p)
            H_LCLs[i,x,j] = pl.calc_H(pl.z_LCL)

tab01_list.append(['broad-min','LCL',275,'%.3E'%5e3,RH,2,'%.3E'%(np.amin(H_LCLs)),'-',0,'-',0,'-'])
tab01_list.append(['broad-max','LCL',400,'%.3E'%1e7,RH,25,'%.3E'%(np.amax(H_LCLs)),'-',0,'-',0,'-'])


# convert to dataframe to easily save to csv file
tab01 = pd.DataFrame(tab01_list,columns=['name','z_ref','T(z_ref) [K]','p_dry(z_ref) [Pa]','RH(z_ref) [ ]','g [m/s]','H_LCL [m]','f_h2 [mol/mol]','f_he [mol/mol]','f_n2 [mol/mol]','f_o2 [mol/mol]','f_co2 [mol/mol]'])
tab01.to_csv('tabs/tab01.csv',index=False)
