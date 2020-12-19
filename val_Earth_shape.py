import numpy as np
import matplotlib.pyplot as plt
import src.drop_prop as drop_prop
from src.planet import Planet

X = np.zeros(5) # composition
X[2] = 0.79 # f_N2  [mol/mol]
X[3] = 0.21 # f_O2  [mol/mol]
T_surf = 290. # [K]
p_surf = 1.01325e5 # [Pa]
RH_surf = 0.75 # []
R_p = 1. # [R_earth]
M_p = 1. # [M_earth]
Earth = Planet(R_p,T_surf,p_surf,X,'h2o',RH_surf,M_p)

thurai_2dvd = np.genfromtxt('data/Thurai2009_2DVD_a_v_d_mm.dat')
thurai_tunnel = np.genfromtxt('data/Thurai2009_windtunnel_a_v_d_mm.dat')
beard1991 = np.genfromtxt('data/Beard1991_a_v_d_mm.dat')

# maximum stable raindrop size
# from Rayleigh-Taylor instability, ℓ_RT = 0.5πa
max_r_Earth = drop_prop.calc_r_max_RT(Earth,np.pi/2.)
n = 100
rs = np.linspace(0.01e-3,max_r_Earth,n)
rat  = np.zeros(n)
for i,r in enumerate(rs):
    rat[i] = Earth.f_rat(r,Earth)

# Pruppacher & Beard 1970
r_lim = np.linspace(0.5,max_r_Earth*1e3,n)
rat_PB70 = 1.030 - 0.062*2*r_lim

# Beard & Chuang (1987)
r_BC87 = np.array([1.,2.,3.,4,5.,6.,7.,8.])/2. # [mm]
rat_BC87 = np.array([0.983,0.928,0.853,0.778,0.708,0.642,0.581,0.521]) # [ ]

# Pruppacher & Pitt (1971)
r_PP71 = np.array([0.0170,0.0305,0.0433,0.0532,0.0620,0.11,0.14,0.15,0.18,0.20,0.25,0.29,0.30,0.35,0.40])*10. # [mm]
rat_PP71 = np.array([0.9993,0.9959,0.9892,0.9813,0.9735,0.916,0.865,0.847,0.795,0.762,0.701,0.664,0.655,0.621,0.583]) # [ ]

plt.plot(rs*1e3, rat, c='indigo', lw=2,label='our model',zorder=8)
plt.plot(r_lim,rat_PB70,c='0.75',lw=2,ls='--',label='Pruppacher & Beard (1970)')
plt.scatter(r_PP71,rat_PP71,marker='o',label='Pruppacher & Pitt (1971)',facecolors='none',edgecolor='0.25',zorder=9)
plt.scatter(r_BC87,rat_BC87,facecolor='0.25',marker='x',label='Beard & Chuang (1987)',zorder=10)
plt.scatter(beard1991[:,0]/2,beard1991[:,1],c='plum',zorder=9,marker='_', label='Beard et al. (1991)')
plt.scatter(thurai_2dvd[:-7,0]/2,thurai_2dvd[:-7,1],c='plum',marker='1',zorder=9,label='Thurai et al. (2009)')
# cut off points beyond r_max
plt.scatter(thurai_tunnel[:,0]/2,thurai_tunnel[:,1],c='plum',zorder=9,marker='1')
plt.xlabel(r'$r_\mathrm{eq}$ [mm]')
plt.ylabel(r'$b/a$ []')
plt.ylim(0,1.05)
plt.legend()
plt.savefig('sfigs/sfig03.pdf',bbox_inches='tight',transparent=True)
plt.close()
