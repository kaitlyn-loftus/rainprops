import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from src.planet import Planet
from src.condensible import Condensible
import src.fall as fall
import src.drop_prop as drop_prop


dir = 'output/validation/'

# Titan shape & velocity validations (vs. Lorenz 1993)
R_gas = 8.31446 #[J/mol/K]
ρ_air_0km = 5.3 # [kg/m3]
ρ_air_10km = 3.67 # [kg/m3]
T_air_0km = 97.2 # [K]
T_air_10km = 85.8 # [K]
μ_Titan = 28e-3 # [kg/mol]

def η_L93(T):
    return 1.718e-5+5.1e-8*(T-273.)

p_air_0km = ρ_air_0km*R_gas/μ_Titan*T_air_0km
p_air_10km = ρ_air_10km*R_gas/μ_Titan*T_air_10km

X = np.zeros(5) # composition
X[2] = 1.
RH_surf = 1e-3 # []
R_Titan = 0.404 # [R_earth]
M_Titan = 0.0225 # [M_earth]
Titan_0km = Planet(R_Titan,T_air_0km,p_air_0km,X,'ch4',RH_surf,M_Titan,is_find_LCL=False)
Titan_10km = Planet(R_Titan,T_air_10km,p_air_10km,X,'ch4',RH_surf,M_Titan,is_find_LCL=False)
Earth_0km = Planet(1.,273.15,1.01325e5,X,'h2o',RH_surf,1.,is_find_LCL=False)

Titan_0km.c.ρ_const = 600.
Titan_10km.c.ρ_const = 600.
Titan_0km.c.σ_const = 0.017
Titan_10km.c.σ_const = 0.017
Titan_0km.η = η_L93(Titan_0km.T)
Titan_10km.η = η_L93(Titan_10km.T)

n = 150
rs = np.linspace(1e-2,9,n)*1e-3/2.
vs = np.zeros((n,2,3))
rats = np.zeros((n,2,3))
for i,r in enumerate(rs):
    vs[i,0,0] = drop_prop.calc_v(r,Titan_0km,drop_prop.calc_rat_Lorenz,drop_prop.calc_C_D_Lorenz)
    vs[i,1,0] = drop_prop.calc_v(r,Titan_0km,Titan_0km.f_rat,Titan_0km.f_C_D)
    vs[i,0,1] = drop_prop.calc_v(r,Titan_10km,drop_prop.calc_rat_Lorenz,drop_prop.calc_C_D_Lorenz)
    vs[i,1,1] = drop_prop.calc_v(r,Titan_10km,Titan_10km.f_rat,Titan_10km.f_C_D)
    rats[i,0,0] = drop_prop.calc_rat_Lorenz(r,Titan_0km,vs[i,1,0])
    rats[i,0,1] = drop_prop.calc_rat_Lorenz(r,Titan_10km,vs[i,1,1])
    rats[i,1,0] = drop_prop.calc_rat_Green(r,Titan_0km)

    if r<3e-3:
        vs[i,0,2] = drop_prop.calc_v(r,Earth_0km,drop_prop.calc_rat_Lorenz,drop_prop.calc_C_D_Lorenz)
        vs[i,1,2] = drop_prop.calc_v(r,Earth_0km,Earth_0km.f_rat,Earth_0km.f_C_D)
    else:
        vs[i,0,2] = None
        vs[i,1,2] = None

np.save(dir+'v_L93',vs)
np.save(dir+'rat_L93',rats)
np.save(dir+'r_L93',rs)

# Titan evaporation validations (vs Graves et al. 2008)
T_surf = 93.515 # [K]
p_surf = 146722 # [Pa]

X = np.zeros(5) # composition
X[2] = 1.
RH_surf = 0.45 # []
R_Titan = 0.404 # [R_earth]
M_Titan = 0.0225 # [M_earth]
Titan_G = Planet(R_Titan,T_surf,p_surf,X,'ch4',RH_surf,M_Titan,is_find_LCL=False,C_D_param='L',rat_param='L',is_Titan=True,is_Graves=True,f_vent_param='m')
Titan = Planet(R_Titan,T_surf,p_surf,X,'ch4',RH_surf,M_Titan,is_find_LCL=False,is_Titan=True,f_vent_param='mh',is_Graves=False)

Titan.c1 = Condensible('n2')
Titan.c2 = Condensible('ch4')
Titan_G.c1 = Condensible('n2')
Titan_G.c2 = Condensible('ch4')


# Thompson+ 1992 values for N2 + CH4

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
    a = calc_a(T)
    b = calc_b(T)
    c = calc_c(T)
    lnγ1 = X2**2*((a+3*b+5*c)-4*(b+4*c)*X2 + 12*c*X2**2)
    return np.exp(lnγ1)

def calc_γ2(T,X1,X2):
    a = calc_a(T)
    b = calc_b(T)
    c = calc_c(T)
    lnγ2 = X1**2*((a-3*b+5*c)+4*(b-4*c)*X1 + 12*c*X1**2)
    return np.exp(lnγ2)

# from Graves 2008 appendix eqs A.26 and A.27

def calc_psatN2(T):
    return 1e5*(10**(3.95-306./T))

def calc_psatCH4(T):
    return 3.4543e9*np.exp(-1145.705/T)

def X1_0(XN2,T,pN2,pCH4):
    XCH4 = 1-XN2
    γN2 = calc_γ1(T,XN2,XCH4)
    γCH4 = calc_γ2(T,XN2,XCH4)
    return (pN2 + pCH4 - γN2*XN2*calc_psatN2(T) - γCH4*XCH4*calc_psatCH4(T))

def Q1_0(QN2,T,X1,r):
    QCH4 = 1 - QN2
    ρ_avg = (QN2/Titan.c1.ρ(T) + QCH4/Titan.c2.ρ(T))**(-1)
    m = 4./3*np.pi*ρ_avg*r**3
    return X1 - (QN2*m)/Titan.c1.μ/((QN2*m)/Titan.c1.μ + (QCH4*m)/Titan.c2.μ)

def calc_eq_XN2CH4(T,pN2,pCH4):
    XN2 = brentq(X1_0,1e-2,1,args=(T,pN2,pCH4))
    XCH4 = 1 - XN2
    return XN2, XCH4

def calc_eq_QN2CH4(T,X1,r):
    QN2 = brentq(Q1_0,1e-2,1,args=(T,X1,r))
    QCH4 = 1 - QN2
    return QN2, QCH4

Titan.z2x4drdz_12(8e3)
Titan.c.type_c = 'combo'
Titan_G.z2x4drdz_12(8e3)
Titan_G.c.type_c = 'combo'


r = 4.75e-3
XN2, XCH4 = calc_eq_XN2CH4(Titan.T,Titan.p_c1,Titan.p_c2)
QN2, QCH4 = calc_eq_QN2CH4(Titan.T,XN2,r)
ρ_avg = (QN2/Titan.c1.ρ(Titan.T) + QCH4/Titan.c2.ρ(Titan.T))**(-1)

m = 4./3*np.pi*ρ_avg*r**3
m1 = QN2*m
m2 = QCH4*m

n = 500
z_start = 8e3
sol = fall.integrate_fall_w_t_T_2condensibles(Titan,m1,m2,z_start)
sol_G = fall.integrate_fall_w_t_T_2condensibles(Titan_G,m1,m2,z_start)

t_end = fall.calc_last_t(sol)
ts = np.linspace(0,t_end,n)
z,m1,m2,T = sol.sol.__call__(ts)

t_end_G = fall.calc_last_t(sol_G)
ts_G = np.linspace(0,t_end_G,n)
z_G,m1_G,m2_G,T_G = sol_G.sol.__call__(ts_G)



r = np.zeros((n,2))
ρs = np.zeros((n,2))
ρs_N2 = np.zeros((n,2))
ρs_CH4 = np.zeros((n,2))
vs = np.zeros((n,2))
zs = np.zeros((n,2))
zs[:,0] = z
zs[:,1] = z_G
Ts = np.zeros((n,2))
Ts[:,0] = T
Ts[:,1] = T_G

for i,Z in enumerate(z):
    m = m1[i] + m2[i]
    Tz = Titan.z2T_Titan(Z)
    ρs[i,0] = (m1[i]/m/Titan.c1.ρ(Tz) + m2[i]/m/Titan.c2.ρ(Tz))**(-1)
    ρs_N2[i,0] = Titan.c1.ρ(Tz)
    ρs_CH4[i,0] = Titan.c2.ρ(Tz)
    r[i,0] = (m/ρs[i,0]*3./4/np.pi)**(1./3)
    Titan.z2x4drdz_12(Z)
    vs[i,0] = drop_prop.calc_v(r[i,0],Titan,Titan.f_rat,Titan.f_C_D)
Titan.z2x4drdz_12(0.)

for i,Z in enumerate(z_G):
    m = m1_G[i] + m2_G[i]
    Tz = Titan_G.z2T_Titan(Z)
    ρs[i,1] = (m1_G[i]/m/Titan_G.c1.ρ(Tz) + m2_G[i]/m/Titan_G.c2.ρ(Tz))**(-1)
    ρs_N2[i,1] = Titan_G.c1.ρ(Tz)
    ρs_CH4[i,1] = Titan_G.c2.ρ(Tz)
    r[i,1] = (m/ρs[i,1]*3./4/np.pi)**(1./3)
    Titan_G.z2x4drdz_12(Z)
    vs[i,1] = drop_prop.calc_v(r[i,1],Titan_G,Titan_G.f_rat,Titan_G.f_C_D)
Titan_G.z2x4drdz_12(0.)

np.save(dir+'zs',zs)
np.save(dir+'T_drops',Ts)
np.save(dir+'rs',r)

np.save(dir+'T_atms',Titan.z2T_Titan(z))

print('t_fall',fall.calc_last_t(sol)/60.,'min')
print('t_fall_G',fall.calc_last_t(sol_G)/60.,'min')

print('T_f',T[-1],'K')
print('T_f_G',T_G[-1],'K')

print('X:',calc_eq_XN2CH4(T[-1],Titan.p_c1,Titan.p_c2))
print('X_G:',calc_eq_XN2CH4(T_G[-1],Titan_G.p_c1,Titan_G.p_c2))
print('r_f',r[-1,0]*1e3,'mm')
print('r_f_G',r[-1,1]*1e3,'mm')
print('v_f',vs[-1,0],'m/s')
print('v_f_G',vs[-1,1],'m/s')
