################################################################
# calculate and make LoWo21 Figures S7-S8, Table S1
# compare results to Graves et al. (2008)
# DOI: 10.1016/j.pss.2007.11.001
################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from src.planet import Planet
from src.condensible import Condensible
import src.fall as fall
import src.drop_prop as drop_prop
import pandas as pd


# code for 2 condensible raindrops is not very elegantly written,
# so some variables and functions here for N2-CH4 that should be implemented from
# condensible.py ....

# N2 + CH4 mixture activity coefficients, used in calc_γ
# from Thompson et al. (1992) Table 1 and eq (7b)
a0 = 0.8096
a1 = -52.07
a2 = 5443.
b0 = -0.0829
b1 = 9.34
c0 = 0.0720
c1 = -6.27
calc_a = lambda T: a0 + a1/T + a2/T**2
calc_b = lambda T: b0 + b1/T
calc_c = lambda T: c0 + c1/T

def calc_γ1(T,X1,X2):
    '''
    activity coefficient of N2 to account for thermodynamic
    effects of liquid N2-CH4 mixture
    (1 -> N2, 2 -> CH4)
    from Thompson et al. (1992) eq(9)
    inputs:
        * T [K] - temperature
        * X1 [mol/mol] - molar concentration of component 1
        * X2 [mol/mol] - molar concentration of component 2
    output:
        * γ1 [ ] - activity coefficient
    '''
    a = calc_a(T)
    b = calc_b(T)
    c = calc_c(T)
    lnγ1 = X2**2*((a+3*b+5*c)-4*(b+4*c)*X2 + 12*c*X2**2)
    return np.exp(lnγ1)

def calc_γ2(T,X1,X2):
    '''
    activity coefficient of CH4 to account for thermodynamic
    effects of liquid N2-CH4 mixture
    (1 -> N2, 2 -> CH4)
    from Thompson et al. (1992) eq (9)
    inputs:
        * T [K] - temperature
        * X1 [mol/mol] - molar concentration of component 1
        * X2 [mol/mol] - molar concentration of component 2
    output:
        * γ2 [ ] - activity coefficient
    '''
    a = calc_a(T)
    b = calc_b(T)
    c = calc_c(T)
    lnγ2 = X1**2*((a-3*b+5*c)+4*(b-4*c)*X1 + 12*c*X1**2)
    return np.exp(lnγ2)

def calc_psatN2(T):
    '''
    calculate saturation pressure of N2 for given T
    from Graves+ (2008) appendix eq (A.26)
    input:
        * T [K] - temperature
    output:
        *p_sat [Pa] - saturation pressure of N2
    '''
    return 1e5*(10**(3.95-306./T))

def calc_psatCH4(T):
    '''
    calculate saturation pressure of CH4 for given T
    from Graves+ (2008) appendix eq (A.27)
    input:
        * T [K] - temperature
    output:
        *p_sat [Pa] - saturation pressure of CH4
    '''
    return 3.4543e9*np.exp(-1145.705/T)


def X1_0(XN2,T,pN2,pCH4):
    '''
    zero function for numerical solver
    to calculate liquid molar concentration of N2 in N2-CH4 mixture
    in equilibrium with air
    following Graves+ (2008) appendix eq (A.23)
    inputs:
        * XN2 [mol/mol] - liquid molar concentration of N2 in N2-CH4 mixture
        * T [K] - local temperature
        * pN2 [Pa] - local partial pressure of N2
        * pCH4 [Pa] - local partial pressure of CH4
    output:
        * difference between air partial pressures of N2 and CH4 and
          drop surface partial pressures of N2 and CH4 [Pa]
    '''
    XCH4 = 1-XN2 # [mol/mol]
    γN2 = calc_γ1(T,XN2,XCH4) # [ ]
    γCH4 = calc_γ2(T,XN2,XCH4) # [ ]
    return (pN2 + pCH4 - γN2*XN2*calc_psatN2(T) - γCH4*XCH4*calc_psatCH4(T)) # [Pa]

def Q1_0(QN2,T,X1,r):
    '''
    zero function for numerical solver
    to calculate liquid mass concentration of N2 in N2-CH4 mixture
    inputs:
        * QN2 [mol/mol] - liquid molar concentration of N2 in N2-CH4 mixture
        * T [K] - local temperature
        * X1 [mol/mol] - liquid molar concentration of N2 in N2-CH4 mixture
        * r [m] - radius
    output:
        * difference between given N2 molar concentration and
          N2 molar concentration calculated from inputted N2 mass concentration [mol/mol]
    '''
    QCH4 = 1 - QN2 # [kg/kg]
    ρ_avg = (QN2/Titan.c1.ρ(T) + QCH4/Titan.c2.ρ(T))**(-1) # [kg/m3], following Graves+ (2008) appendix eq (A.7)
    m = 4./3*np.pi*ρ_avg*r**3 # [kg]
    return X1 - (QN2*m)/Titan.c1.μ/((QN2*m)/Titan.c1.μ + (QCH4*m)/Titan.c2.μ) # [mol/mol]

def calc_eq_XN2CH4(T,pN2,pCH4):
    '''
    calculate liquid molar concentration of N2 and CH4 in N2-CH4 mixture
    in equilibrium with air
    following Graves+ (2008) appendix eq (A.23)
    inputs:
        * T [K] - local temperature
        * pN2 [Pa] - local partial pressure of N2
        * pCH4 [Pa] - local partial pressure of CH4
    outputs:
        * XN2 [mol/mol] - liquid molar concentration of N2
        * XCH4 [mol/mol] - liquid molar concentration of CH4
    '''
    # solve numerically for XN2 that is in equilibrium with local conditions
    XN2 = brentq(X1_0,1e-2,1,args=(T,pN2,pCH4)) # [mol/mol]
    # 1 = XCH4 + XN2
    XCH4 = 1 - XN2
    return XN2, XCH4

def calc_eq_QN2CH4(T,X1,r):
    '''
    to calculate liquid mass concentration of N2 and CH4 in N2-CH4 mixture
    inputs:
        * T [K] - local temperature
        * X1 [mol/mol] - liquid molar concentration of N2 in N2-CH4 mixture
        * r [m] - radius
    output:
        * QN2 [kg/kg] - liquid mass concentration of N2 in N2-CH4 mixture
        * QCH4 [kg/kg] - liquid mass concentration of CH4 in N2-CH4 mixture
    '''
    QN2 = brentq(Q1_0,1e-2,1,args=(T,X1,r))
    QCH4 = 1 - QN2
    return QN2, QCH4
###################################################################

X = np.zeros(5) # composition
X[2] = 1. # f_N2 [mol/mol]
# following Huygens data
T_surf = 93.515 # [K]
p_surf = 146722 # [Pa]
RH_surf = 0.45 # [ ]
R_Titan = 0.404 # [R_earth]
M_Titan = 0.0225 # [M_earth]
# make Planet for Graves+ (2008) method
Titan_G = Planet(R_Titan,T_surf,p_surf,X,'ch4',RH_surf,M_Titan,is_find_LCL=False,C_D_param='L',rat_param='L',is_Titan=True,is_Graves=True,f_vent_param='m')
# make Planet for LoWo21 method
Titan = Planet(R_Titan,T_surf,p_surf,X,'ch4',RH_surf,M_Titan,is_find_LCL=False,is_Titan=True,f_vent_param='mh',is_Graves=False)

# set up multiple condensibles for both methods
Titan.c1 = Condensible('n2')
Titan.c2 = Condensible('ch4')
Titan_G.c1 = Condensible('n2')
Titan_G.c2 = Condensible('ch4')

# set up Planets to integrate
# set initial z to 8 km following Graves+ (2008)
# change condensible type to 'combo' for N2-CH4 raindrops to work
Titan.z2x4drdz_12(8e3)
Titan.c.type_c = 'combo'
Titan_G.z2x4drdz_12(8e3)
Titan_G.c.type_c = 'combo'

r = 4.75e-3 # [m] r0 following Graves+ (2008)
# initial molar concentration of N2-CH4 raindrop
XN2, XCH4 = calc_eq_XN2CH4(Titan.T,Titan.p_c1,Titan.p_c2) # [mol/mol, mol/mol], [f_N2, f_CH4]
# initial mass concentration of N2-CH4 raindrop
QN2, QCH4 = calc_eq_QN2CH4(Titan.T,XN2,r) # [kg/kg, kg/kg], [q_N2, q_CH4]
# initial N2-CH4 raindrop density
ρ_avg = (QN2/Titan.c1.ρ(Titan.T) + QCH4/Titan.c2.ρ(Titan.T))**(-1) # [kg/m3]
# total inital raindrop mass
m = 4./3*np.pi*ρ_avg*r**3 # [kg]
# initial raindrop N2 mass
m1 = QN2*m # [kg]
# initial raindrop N2 mass
m2 = QCH4*m # [kg]
# initial z following Graves+ (2008)
z_start = 8e3 # [m]

# integrate raindrop falling
sol = fall.integrate_fall_w_t_T_2condensibles(Titan,m1,m2,z_start) # our method
sol_G = fall.integrate_fall_w_t_T_2condensibles(Titan_G,m1,m2,z_start) # Graves+ (2008) reproduction

# number of z points for plot
n = 500

# outputs of integration for our model
t_end = fall.calc_last_t(sol) # [s]
ts = np.linspace(0,t_end,n) # [s]
z,m1,m2,T = sol.sol.__call__(ts) # [m, kg, kg, K]

# outputs of integration for Graves+ (2008) reproduction
t_end_G = fall.calc_last_t(sol_G) # [s]
ts_G = np.linspace(0,t_end_G,n) # [s]
z_G,m1_G,m2_G,T_G = sol_G.sol.__call__(ts_G) # [m, kg, kg, K], each var is an array of size n

# load data from Graves+ (2008) figures
g08_zTdrop = np.genfromtxt('data/Graves2008_z_v_Tdrop.csv',delimiter=',')
g08_zr = np.genfromtxt('data/Graves2008_z_v_r_mm.csv',delimiter=',')

# make plot of z vs T, Figure S8
plt.plot(Titan.z2T_Titan(z),z,c='0.75',lw=2,label=r'$T_\mathrm{air}$')
plt.plot(T,z,c='indigo',lw=2,label=r'$T_\mathrm{drop}$, this work')
plt.plot(T_G,z_G,c='plum',lw=2,ls='--',label=r'$T_\mathrm{drop}$, Graves et al. (2008) reproduction')
plt.scatter(g08_zTdrop[:,0],g08_zTdrop[:,1],color='plum',marker='1',label=r'$T_\mathrm{drop}$, Graves et al. (2008)',zorder=10)

plt.xlabel(r'$T$ [K]')
plt.ylabel(r'$z$ [m]')
plt.legend()
plt.ylim(0,8e3)
plt.savefig('sfigs/sfig08.pdf',transparent=True,bbox_inches='tight')
plt.close()


# set up empty arrays for z dependent outputs
# 2 columns for 2 calculation methods (ours; Graves+, 2008)
r = np.zeros((n,2)) # [m] radius
ρs = np.zeros((n,2)) # [kg/m3] density
vs = np.zeros((n,2)) # [m/s] raindrop velocity

# for our method
for i,Z in enumerate(z):
    m = m1[i] + m2[i] # [kg] total raindrop mass
    Tz = Titan.z2T_Titan(Z) # [K] air T
    ρs[i,0] = (m1[i]/m/Titan.c1.ρ(Tz) + m2[i]/m/Titan.c2.ρ(Tz))**(-1) # [kg/m3] raindrop radius
    r[i,0] = (m/ρs[i,0]*3./4/np.pi)**(1./3) # [m] raindrop radius
    Titan.z2x4drdz_12(Z) # set z-dependent variables to z = Z
    vs[i,0] = drop_prop.calc_v(r[i,0],Titan,Titan.f_rat,Titan.f_C_D) # [m/s] raindrop velocity

# for Graves+ (2008) method
for i,Z in enumerate(z_G):
    m = m1_G[i] + m2_G[i] # [kg] total raindrop mass
    Tz = Titan_G.z2T_Titan(Z) # [K] air T
    ρs[i,1] = (m1_G[i]/m/Titan_G.c1.ρ(Tz) + m2_G[i]/m/Titan_G.c2.ρ(Tz))**(-1) # [kg/m3] raindrop radius
    r[i,1] = (m/ρs[i,1]*3./4/np.pi)**(1./3) # [m] raindrop radius
    Titan_G.z2x4drdz_12(Z) # set z-dependent variables to z = Z
    vs[i,1] = drop_prop.calc_v(r[i,1],Titan_G,Titan_G.f_rat,Titan_G.f_C_D) # [m/s] raindrop velocity

# calculate liquid mole concentration of raindrop at surface
Titan_G.z2x4drdz_12(0.) # reset z to surface
Titan.z2x4drdz_12(0.) # reset z to surface
X = calc_eq_XN2CH4(T[-1],Titan.p_c1,Titan.p_c2) # [mol/mol, mol/mol], [f_N2, f_CH4]
X_G = calc_eq_XN2CH4(T_G[-1],Titan_G.p_c1,Titan_G.p_c2) # [mol/mol, mol/mol], [f_N2, f_CH4], for Graves+ (2008) method

# make Table S1
stab01_list = [] # set up list for outputting table results to compare to Graves+ (2008)
stab01_list.append(['r(z=0) [mm]',r[-1,1]*1e3,r[-1,0]*1e3])
stab01_list.append(['t_fall [min]',fall.calc_last_t(sol_G)/60.,fall.calc_last_t(sol)/60.])
stab01_list.append(['v(z=0) [m/s]',vs[-1,1],vs[-1,0]])
stab01_list.append(['fCH4(z=0) [m/s]',X_G[1],X[1]])
stab01_list.append(['fN2(z=0) [m/s]',X_G[0],X[0]])
stab01_list.append(['Tdrop(z=0) [m/s]',T_G[-1],T[-1]])
# convert to dataframe to easily save to csv file
tabs01 = pd.DataFrame(stab01_list,columns=['property','G08 reproduction','this work'])
tabs01.to_csv('stabs/stab01.csv',index=False)

# make plot of z vs. r, Figure S7
plt.plot(r[:,0]*1e3,z,c='indigo',lw=2,label='this work')
plt.plot(r[:,1]*1e3,z,c='plum',lw=2,ls=':',label='Graves et al. (2008)\nreproduction')
plt.scatter(g08_zr[:,0],g08_zr[:,1],color='plum',marker='1',zorder=10,label='Graves et al. (2008)')
plt.xlabel(r'$r_\mathrm{eq}$ [mm]')
plt.ylabel(r'$z$ [m]')
plt.xlim(0,5)
plt.ylim(0,8e3)
plt.legend()
plt.savefig('sfigs/sfig07.pdf',transparent=True,bbox_inches='tight')
plt.close()
