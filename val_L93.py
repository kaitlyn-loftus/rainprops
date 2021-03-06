################################################################
# calculate and make LoWo21 Figures S5-S6
# compare results to Lorenz (1993)
# DOI: 10.1016/0032-0633(93)90048-7
################################################################
import src.drop_prop as drop_prop
from src.planet import Planet
import matplotlib.pyplot as plt
import numpy as np

R_gas = 8.31446 #[J/mol/K]
# Titan atm properties from Lorenz (1993) Table A1
ρ_air_0km = 5.3 # [kg/m3]
ρ_air_10km = 3.67 # [kg/m3]
T_air_0km = 97.2 # [K]
T_air_10km = 85.8 # [K]
μ_Titan = 28e-3 # [kg/mol], assume pure N2

def η_L93(T):
    '''
    calculate dynamic viscosity as a function of temperature for N2 atm
    from Lorenz (1993) eq (A1)
    input:
        * T [K] - temperature
    output:
        * η [Pa s] - dynamic viscosity
    '''
    return 1.718e-5+5.1e-8*(T-273.)

# convert density to pressure
p_air_0km = ρ_air_0km*R_gas/μ_Titan*T_air_0km # [Pa]
p_air_10km = ρ_air_10km*R_gas/μ_Titan*T_air_10km # [Pa]

# set up planet objects for calculations
# Titan @ z = 0, 10 km and Earth @ z = 0 km
X = np.zeros(5) # composition
X[2] = 1.
RH_surf = 1e-3 # [ ]
R_Titan = 0.404 # [R_earth]
M_Titan = 0.0225 # [M_earth]
Titan_0km = Planet(R_Titan,T_air_0km,p_air_0km,X,'ch4',RH_surf,M_Titan,is_find_LCL=False)
Titan_10km = Planet(R_Titan,T_air_10km,p_air_10km,X,'ch4',RH_surf,M_Titan,is_find_LCL=False)
Earth_0km = Planet(1.,273.15,1.01325e5,X,'h2o',RH_surf,1.,is_find_LCL=False)

# set up planets to follow Lorenz (1993)
# for raindrop liquid condensible density, surface tension, and dynamic viscosity
Titan_0km.c.is_lorenz = True
Titan_10km.c.is_lorenz = True
Titan_0km.c.ρ_const = 600. # [kg/m3]
Titan_10km.c.ρ_const = 600. # [kg/m3]
Titan_0km.c.σ_const = 0.017 # [N/m]
Titan_10km.c.σ_const = 0.017 # [N/m]
Titan_0km.η = η_L93(Titan_0km.T) # [Pa s]
Titan_10km.η = η_L93(Titan_10km.T) # [Pa s]

n = 150 # number of radii to calculate v and rat for
rs = np.linspace(1e-2,9,n)*1e-3/2. # [m] raindrop radius
vs = np.zeros((n,2,3)) # [m/s] raindrop velocity
rats = np.zeros((n,2,3)) # [ ] shape ratios b/a
# calculate vs and ratios for different methods and planetary conditions
for i,r in enumerate(rs):
    vs[i,0,0] = drop_prop.calc_v(r,Titan_0km,drop_prop.calc_rat_Lorenz,drop_prop.calc_C_D_Lorenz)
    vs[i,1,0] = drop_prop.calc_v(r,Titan_0km,Titan_0km.f_rat,Titan_0km.f_C_D)
    vs[i,0,1] = drop_prop.calc_v(r,Titan_10km,drop_prop.calc_rat_Lorenz,drop_prop.calc_C_D_Lorenz)
    vs[i,1,1] = drop_prop.calc_v(r,Titan_10km,Titan_10km.f_rat,Titan_10km.f_C_D)
    rats[i,0,0] = drop_prop.calc_rat_Lorenz(r,Titan_0km,vs[i,1,0])
    rats[i,0,1] = drop_prop.calc_rat_Lorenz(r,Titan_10km,vs[i,1,1])
    rats[i,1,0] = drop_prop.calc_rat_Green(r,Titan_0km)

    if r<3e-3: # only include Earth values up to when Lorenz does
        vs[i,0,2] = drop_prop.calc_v(r,Earth_0km,drop_prop.calc_rat_Lorenz,drop_prop.calc_C_D_Lorenz)
        vs[i,1,2] = drop_prop.calc_v(r,Earth_0km,Earth_0km.f_rat,Earth_0km.f_C_D)
    else:
        vs[i,0,2] = None
        vs[i,1,2] = None

# load in Lorenz (1993) values
L93 = np.genfromtxt('data/Lorenz1993_v_d.csv',delimiter=',',skip_header=1)

# make plot of v vs r, Figure S6
plt.plot(rs*1e3,vs[:,1,0],label='this work',c='indigo')
plt.plot(rs*1e3,vs[:,0,0],label='Lorenz (1993) reproduction',c='plum')
plt.scatter(L93[:,0]/2.,L93[:,1],color='plum',marker='1',zorder=10,label='Lorenz (1993)')
plt.plot(rs*1e3,vs[:,0,1],c='plum',ls='--')
plt.plot(rs*1e3,vs[:,1,1],c='indigo',ls='--')
plt.scatter(L93[:,0]/2.,L93[:,2],color='plum',marker='1',zorder=10)
plt.plot(rs*1e3,vs[:,0,2],c='plum',ls=':')
plt.plot(rs*1e3,vs[:,1,2],c='indigo',ls=':')
plt.scatter(L93[:,0]/2.,L93[:,3],color='plum',marker='1',zorder=10)
plt.xlabel(r'$r_\mathrm{eq}$ [mm]')
plt.ylabel(r'$v_\mathrm{T}$ [m s$^{-1}$]')
plt.xlim(0,4.55)
plt.ylim(0,9.5)
# legend
plt.plot([-1,-2],[-1,-2],color='0.75',ls=':',label=r'Earth (H$_2$O), 0 km')
plt.plot([-1,-2],[-1,-2],color='0.75',ls='--',label=r'Titan (CH$_4$-N$_2$), 10 km')
plt.plot([-1,-2],[-1,-2],color='0.75',label=r'Titan (CH$_4$-N$_2$), 0 km')
plt.legend()
plt.savefig('sfigs/sfig06.pdf',transparent=True,bbox_inches='tight')
plt.close()

# make plot of shape ratio vs r, Figure S5
plt.plot(rs*1e3,rats[:,1,0],label='this work',c='indigo')
plt.plot(rs*1e3,rats[:,0,0],label='Lorenz (1993) reproduction',c='plum')
plt.ylabel(r'$b/a$ [ ]')
plt.xlabel(r'$r_\mathrm{eq}$ [mm]')
plt.legend()
plt.ylim(0,1.01)
plt.savefig('sfigs/sfig05.pdf',transparent=True,bbox_inches='tight')
plt.close()
