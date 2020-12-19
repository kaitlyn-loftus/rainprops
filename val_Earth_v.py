import src.drop_prop as drop_prop
import numpy as np
import matplotlib.pyplot as plt
from src.planet import Planet


X = np.zeros(5) # composition
X[2] = 0.8 # f_N2  [mol/mol]
X[3] = 0.2
T_surf = 293.15 # [K]
p_surf = 1.01325e5 # [Pa]
RH_surf = 0.5 # []
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
pl = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)
pl.z2x4drdz(0.)

n = 100
rs = np.logspace(-6,np.log10(7.25e-3/2.),n)
v_B = np.zeros(n)
v_Lorenz = np.zeros(n)
v_Lorenz2 = np.zeros(n)
v_Holzer = np.zeros(n)
v_Salman = np.zeros(n)
v_Ganser_Newton = np.zeros(n)
v_sphere = np.zeros(n)
v_Loth = np.zeros(n)

for i,r in enumerate(rs):
    v_B[i] = drop_prop.calc_v_Earth_Beard(r,pl)
    v_Lorenz[i] = drop_prop.calc_v(r,pl,pl.f_rat,drop_prop.calc_C_D_Lorenz)
    v_Lorenz2[i] = drop_prop.calc_v(r,pl,drop_prop.calc_rat_Lorenz,drop_prop.calc_C_D_Lorenz)
    v_Holzer[i] = drop_prop.calc_v(r,pl,pl.f_rat,drop_prop.calc_C_D_Holzer)
    v_Salman[i] = drop_prop.calc_v(r,pl,pl.f_rat,drop_prop.calc_C_D_Salman)
    v_Ganser_Newton[i] = drop_prop.calc_v(r,pl,pl.f_rat,drop_prop.calc_C_D_Ganser_Newton)
    v_sphere[i] = drop_prop.calc_v(r,pl,pl.f_rat,drop_prop.calc_C_D_sphere)
    v_Loth[i] = drop_prop.calc_v(r,pl,pl.f_rat,drop_prop.calc_C_D_Loth)


gunn49 = np.genfromtxt('data/gunn1949_r_m_v_mpers.csv',delimiter=',')
best50 = np.genfromtxt('data/best1950.csv',delimiter=',')

plt.plot(rs*1e3,v_Loth,c='indigo',lw=3,label='this work')
plt.plot(rs*1e3,v_B,c='plum',ls='--',lw=2,label='Beard (1976)')
plt.scatter(gunn49[:,0]*1e3,gunn49[:,1],c='plum',label='Gunn & Kinzer (1949)',marker='1',zorder=10)
plt.scatter(best50[:,0]/2.,best50[:,1],c='plum',label='Best (1950)',marker='+',zorder=10)
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.ylabel(r'$v_\mathrm{T}$ [m s$^{-1}$]')
plt.xlabel(r'$r_\mathrm{eq}$ [mm]')
plt.savefig('sfigs/sfig04.pdf',transparent=True,bbox_inches='tight')
plt.close()
