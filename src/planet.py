################################################################
# class for storing and calculating planetary attributes
################################################################
import numpy as np
import src.condensible as condensible
import src.drop_prop as drop_prop
from scipy.optimize import brentq
import src.huygens as huygens

# CONSTANTS
# gravitational constant
G = 6.674e-11 # [Nm2/kg2]
# ideal gas constant
R_gas = 8.31446 #[J/mol/K]
# Avogadro's number
N_A = 6.022141e23 # [particles/mol]

# EARTH SPECIFIC VALUES
M_earth = 5.9721986e24 # [kg]
R_earth = 6.371e6 # [m]

# read in gas data
gas_properties = np.genfromtxt('./data/gas_properties.csv',delimiter=',',names=True)


def T_transition_moist0(T,pl):
    '''
    calculate temperature at which RH = 1 and hit LCL
    assumes dry adiabat below LCL,
    assuming condensible species is well-mixed below LCL
    inputs:
        * T [K] - local temperature
        * pl [Planet]
    output:
        * difference between local pH2O and saturated pH2O at given T [Pa]
    '''
    kappa = pl.R_sp_air/pl.c_p
    return pl.f_c*pl.p_surf*(T/pl.T_surf)**(1./kappa) - pl.c.p_sat(T)

class Planet:
    '''
    Planet class
    attributes (not quite exhaustive):
        general
            * R [m] - planet radius
            * M [kg] - planet mass
            * g [m/s2] - surface gravity
            * c [Condensible object] - contains condensible properties for species assigned to planet
            * f_c [mol/mol] - molar concentration of condensible species below LCL (assumed to be constant)
            * μ_dry [kg/mol] - average dry atmospheric molar mass
            * μ_air [kg/mol] - average atmospheric molar mass (includes condensible species)
            * R_sp_air [J/kg/K] - specific gas constant for air (includes condensible species)
            * c_p [J/kg/K] - specific heat capacity for air (includes condensible species)
            * is_Cc [boolean] - whether to account for Cunningham-Stokes correction to drag, default is no (minimal difference if True)
        surface values ("surface" is equivalent to where z=0)
            * T_surf [K] - surface temperature
            * p_surf [Pa] - surface pressure total (will be adjusted later to include condensible vapor)
            * RH_surf [ ] - relative humidity
        LCL values (may be equal to surface values)
            * T_LCL [K]
            * z_LCL [m]
        local values (set by z from z2x4drdz)
            * z, T, p, ρ, K, D_c, RH, η
    methods:
        * __init__ - constructor
        methods to relate atmospheric variables to (2) each other
            * z2T
            * T2p
            * z2p
            * pT2ρ
            * pT2RH
            * z2x4drdz
        methods to calculate properties of air
            * calc_D_c
            * calc_D_c_base
            * calc_Ω_D
            * calc_Ω_η
            * calc_K
            * calc_c_p
            * calc_η
            * calc_H
            specific to Titan scenario (not generalized 😿 presently)
                * z2x4drdz_12
                taken from Graves et al. (2008)
                    * calc_ηG
                    * calc_KG
                    * calc_D_c1G
                    * calc_D_c2G
        methods to calculate drop properties
            * f_C_D
            * f_rat
            * f_vent
    '''
    def __init__(self,Rp_earth,T_surf,p_surf_dry,atm_comp,cndsbl_species,RH_surf,
                 Mp_earth=None,g_force=None,C_D_param='K8',rat_param='G',
                 f_vent_param='mh',is_find_LCL=True,is_Titan=False,is_Graves=False,is_add_pc2psurf=True,is_Cc=False):
        '''
        constructor for Planet class
        initialize Planet object
        inputs:
            * Rp_earth [Earth radii] - planet radius
            * T_surf [K] - temperature at input z
            * p_surf [Pa] -  pressure at input z
            * atm_comp [volume mixing ratio] - dry atmospheric composition array
                                               [H2, He, N2, O2, CO2]
            * cndsbl_species [string] - condensible species, presently only h2o & ch4
                                        as robust options
            * RH_surf [] - relative humidity value at input z, between 0 and 1
            * (optional) Mp_earth [Earth masses] - planet mass
            * (optional) g_force [m/s2] - preset surface gravity value
            * (optional) C_D_parm [string] - sets parameterization used to calculate drag coefficient, see drop_prop, default is described in LoWo eqs (6-7)
            * (optional) rat_param [string] - sets method for calculating raindrop axis ratio (b/a), see drop_prop, default is to use method of Green (1975), LoWo eq (2)
            * (optional) f_vent_param [string] - sets parameterization used to calculate ventilation factors for molecular and heat transport, see drop_prop, default is to use separate methods as described in LoWo eqs (12-14)
            * (optional) is_find_LCL [boolean] - whether or not to attempt to numerically solve for LCL, default is True, set to false when don't want to do a calculation with altitude
            * (optional) is_Titan [boolean] - whether Planet is attempting to mimic Titan, default is False, if True will define functions for calculating atmospheric properties from interpolated Huygens probe data
            * (optional) is_Graves [boolean] - whether trying to mimic Graves et al. (2008)
            * (optional) is_add_pc2psurf [boolean] - whether to add condensible pressure to inputted p_input, alternative is to keep p_input as p_total and substract condensible pressure to get dry pressure, default is True
            * (optional) is_Cc [boolean] - whether or not Cunningham-Stokes correction is used for v calculations, default is False

        '''
        self.R = Rp_earth*R_earth # [m] planetary radius
        # set planetary surface gravity
        if g_force==None:
            # set planetary mass
            # if no input, scale mass from radius using Valencia et al. 2006 scaling
            if Mp_earth==None:
                # Valencia et al. 2006 scaling
                conversion_V2006 = M_earth/R_earth**(1./0.27) # [whatever makes below eq have correct units]
                self.M = conversion_V2006*self.R**(1./0.27) # [kg]
            else:
                self.M = Mp_earth*M_earth # [kg]
            self.g = G*self.M/self.R**2 # [m/s2] surface gravity
        else:
            # force g to be predetermined value
            self.g = g_force # [m/s2] surface gravity
            # (re)set planet mass to be ''correct'' value from R and g
            self.M = self.g/G*self.R**2 # [kg]

        self.T_surf = T_surf # [K] surface temperature
        self.p_surf = p_surf_dry # [Pa] surface pressure

        # confirm atmospheric composition sums to 1 to ~ machine error
        # and there are 5 atmospheric gases with concentrations
        # if not, terminate proceedings
        if abs(np.sum(atm_comp) - 1.) > 1e-9:
            raise Exception('atmospheric composition does not sum to 1')
        if atm_comp.shape[0] != 5:
            raise Exception('atmospheric composition does not have the correct # of component dry gases')

        # set up condensible object
        self.c = condensible.Condensible(cndsbl_species)


        self.RH_surf = RH_surf # []
        # add condensible gas to p_set, depending on is_add_pc2psurf input
        if is_add_pc2psurf:
            self.p_surf += self.RH_surf*self.c.p_sat(self.T_surf) # [Pa]
        self.f_c = self.RH_surf*self.c.p_sat(self.T_surf)/self.p_surf # [mol/mol]

        # whether to use p(z), T(z), f_CH4(z) from interpolated Huygens data
        if is_Titan:
            self.z2p_Titan, self.z2T_Titan, self.z2fCH4_Titan = huygens.getz2xfunctions()
            # whether to use Graves et al. (2008) methods
            self.is_Graves = is_Graves


        # set up composition indices array
        X_non0i = []
        # get which species are present in given atmosphere
        for i,X_i in enumerate(atm_comp):
            if X_i!=0:
                X_non0i.append(i)
        # also include condensible species
        X_non0i.append(self.c.gas_i)
        # number of gases
        self.n_gas = len(X_non0i) # [int]
        # set up arrays for calculating composition dependent values
        self.X_μ = np.zeros(self.n_gas) # molar mass [kg/mol]
        self.X_cp = np.zeros((self.n_gas,4)) # values to calculate specific heat at constant pressure
        self.X_η = np.zeros((self.n_gas,3)) # values to calculate dynamic viscosity
        self.X_d = np.zeros(self.n_gas) # molecule diameter
        # initialize arrays for D_c calculations
        self.X_ep_kB = np.zeros(self.n_gas) # Lennard-Jones energy / Boltzmann constant, to eventually calculate D_c, K
        self.X_ep_kB_avg = np.zeros(self.n_gas)
        self.X_D_c_base = np.zeros(self.n_gas)

        self.X = np.zeros(self.n_gas) # composition, molar concentration [mol/mol]
        self.X[-1] = self.f_c # set composition for condensible species
        # fill in relevant values from gas_properties data
        for i,Xi in enumerate(X_non0i):
            self.X_μ[i] = gas_properties['molar_mass'][Xi]
            self.X_cp[i,0] = gas_properties['c_p_0'][Xi]
            self.X_cp[i,1] = gas_properties['c_p_1'][Xi]
            self.X_cp[i,2] = gas_properties['c_p_2'][Xi]
            self.X_cp[i,3] = gas_properties['c_p_3'][Xi]
            self.X_η[i,0] = gas_properties['eta_T0'][Xi]
            self.X_η[i,1] = gas_properties['eta_C'][Xi]
            self.X_η[i,2] = gas_properties['eta_eta0'][Xi]
            self.X_d[i] = gas_properties['d'][Xi]
            self.X_ep_kB[i] = gas_properties['ep_kB'][Xi]
            if i!=self.n_gas-1:
                self.X[i] = atm_comp[Xi]*(1-self.f_c)

        # calculate base component (non-pT dependent part) of mass diffusion coefficient D_c
        self.calc_D_c_base()

        self.μ_dry = np.sum(self.X_μ[:-1]*self.X[:-1]) # [kg/mol] average dry atmospheric molar mass
        self.μ_air = np.sum(self.X_μ*self.X) # [kg/mol] average atmospheric molar mass
        self.R_sp_air = R_gas/self.μ_air # [J/kg/K] specific gas constant for air
        self.c_p = self.calc_c_p(T_surf) # [J/kg/K] specific heat for air, assumed to be constant from T_surf value
        # whether to numerically solve for LCL
        if is_find_LCL:
            if self.RH_surf!=1: # don't bother to solve if inputted LCL
            # adjust T-search range depending on condensible species
                if cndsbl_species=='h2o':
                    self.T_LCL = brentq(T_transition_moist0,self.T_surf,150,self) # [K]
                elif cndsbl_species=='ch4':
                    self.T_LCL = brentq(T_transition_moist0,self.T_surf,50,self) # [K]
                self.z_LCL = self.T2z(self.T_LCL) # [m]
            else:
                self.T_LCL = self.T_surf # [K]
                self.z_LCL = 0. # [m] if inputted LCL, set z=0 to LCL

        # set functions for C_D, axis ratio, and ventilation factors
        self.f_C_D = drop_prop.assign_f_C_D(C_D_param)
        self.f_rat = drop_prop.assign_f_rat(rat_param)
        self.f_vent = drop_prop.assign_f_vent(f_vent_param)

        self.z2x4drdz(0.) # set all z-dependent values to z=0

        if self.c.type_c=='h2o':
            self.c.calc_ρ_h2o(self.T) # set H2O density to T-dependent value at z=0
        # calculate Cunningham-Stokes correction?
        self.is_Cc = is_Cc



    def z2T(self,z):
        '''
        calculate temperature from altitude
        assuming dry adiabat, neglecting T dependence of c_p within atm,
        assuming condensible species is well-mixed below cloud base
        inputs:
            * self
            * z [m] - altitude
        output:
            * T [K] - temperature
        '''
        return self.T_surf - self.g/self.c_p*z

    def T2z(self,T):
        '''
        calculate z from T
        assuming dry adiabat, neglecting T dependence of c_p within atm,
        assuming condensible species is well-mixed below cloud base
        inputs:
            * self
            * T [K] - temperature
        output:
            * z [m] - altitude
        '''
        return (self.T_surf-T)*self.c_p/self.g

    def T2p(self,T):
        '''
        calculate pressure from temperature
        assuming dry adiabat, neglecting T dependence of c_p within atm,
        assuming condensible species is well-mixed below cloud base
        inputs:
            * self
            * T [K] - temperature
        output:
            p [Pa] - pressure
        '''
        return self.p_surf*(T/self.T_surf)**(self.c_p/self.R_sp_air)

    def z2p(self,z):
        '''
        calculate p from z
        assuming dry adiabat, neglecting T dependence of c_p within atm,
        assuming, assuming condensible species is well-mixed below
        cloud base, assuming hydrostatic equilibrium
        inputs:
            * self
            * z [m] - altitude
        output:
            * p [Pa] - pressure
        '''
        T = self.T_surf - self.g/self.c_p*z
        return self.p_surf*(T/self.T_surf)**(self.c_p/self.R_sp_air)

    def pT2ρ(self,p,T):
        '''
        calculate air density ρ from p and T
        assuming dry adiabat, assuming condensible species is well-mixed below
        cloud base
        inputs:
            p [Pa] - pressure
            T [K] - temperature
        output:
            * ρ [kg/m3]
        '''
        return p/T/self.R_sp_air

    def pT2RH(self,p,T):
        '''
        calculate relative humidity from p and T
        assuming dry adiabat, assuming condensible species is well-mixed below
        cloud base
        inputs:
            * self
            * p [Pa] - pressure
            * T [K] - temperature
        output:
            * RH [Pa/Pa] - relative humidity
        '''
        return self.f_c*p/self.c.p_sat(T)


    def calc_D_c(self,T=None,p=None):
        '''
        calculate mass diffusion coefficient for condensible gas in air from
        eq (11-3.2) pg 549 Reid+ (1977)
        inputs:
            * self
            * (optional) T [K]
            * (optional) p [Pa]
        output:
            * D_c [m2/s] - mass diffusion coefficient for condensible gas in air
        '''
        if T==None:
            T = self.T
        if p==None:
            p = self.p

        D_c = 0
        # combine for mixture following Fairbanks & Wilke (1950)
        # except also include self-diffusion term
        for i,X in enumerate(self.X):
            D_Xc = self.X_D_c_base[i]*T**1.5/(p/1.01325e5)/self.calc_Ω_D(T/self.X_ep_kB_avg[i])
            D_c += X/D_Xc
        return D_c**(-1)*1e-4

    def calc_D_c_base(self):
        '''
        calculate the base of the composition dependent values for
        calculating D_c
        input:
            * self
        '''
        for i,X in enumerate(self.X):
            d_avg = 0.5*(self.X_d[i] + self.X_d[-1]) # avg according to Reid+ (1977), eq (11-3.4), pg 549
            self.X_D_c_base[i] = 1.858e-3*(1/(self.X_μ[i]*1e3)+1/(self.X_μ[-1]*1e3))**0.5/d_avg**2
            self.X_ep_kB_avg[i] = (self.X_ep_kB[i]*self.X_ep_kB[-1])**0.5 # avg according to Reid+ 1977, eq 11-3.5, pg 549

    def calc_Ω_D(self,T_star):
        '''
        calculate diffusion collision integral
        following eq (11-3.6) Reid+ (1977) (pg 549-550)
        inputs:
            * self
            * T_star [] - kT/ϵ, (ϵ=Lennard-Jones energy)
        outputs:
            * Ω [] - diffusion collision integral
        '''
        Ω = 1.06036/T_star**0.15610 + 0.19300/np.exp(0.47635*T_star) + 1.03587/np.exp(1.52996*T_star) + 1.76474/np.exp(3.89411*T_star)
        return Ω

    def calc_Ω_η(self,T_star):
        '''
        calculate collision integral
        following eq (9-4.3) Reid+ (1977) (pg 396)
        inputs:
            * self
            * T_star [] - kT/ϵ, (ϵ=Lennard-Jones energy)
        outputs:
            * Ω [] - collision integral
        '''
        Ω = 1.16145/T_star**0.14874 + 0.52487/np.exp(0.77320*T_star) + 2.16178/np.exp(2.43787*T_star)
        return Ω


    def calc_K(self,T=None):
        '''
        calculate thermal conductivity of air (K) at local z level
        inputs:
            * self
            * (optional) T - temperature
        output:
            * K [W/K] - thermal conductivity at local z level
        '''
        if T==None:
            T = self.T
        # calc η for individual gases given T based on Reid+ (1977) eq 9-3.9
        η_i = 1e-7*26.69*(self.X_μ*1e3*T)**0.5/self.X_d**2/self.calc_Ω_η(T/self.X_ep_kB)
        θ = T/1000.
        c_p_i = (self.X_cp[:,0] + self.X_cp[:,1]*θ + self.X_cp[:,2]*θ**2 + self.X_cp[:,3]*θ**3)*1000.
        R_i = R_gas/self.X_μ
        c_v_i = c_p_i - R_i
        # calc K for individual gases based on Eucken correction
        # Reid+ (1977) eq (10-3.3) pg 473
        # Eucken less theoretically rigorous but works better vs experiments
        # for other simple methods
        # pg 480, 495
        K_i = η_i*c_v_i/4.*(9*c_p_i/c_v_i-5.)
        K = 0.
        # weight K_i in mixture by Wilke's approach + Wassiljewa eq
        # Reid+ (1977) pg 508, eq (10-6.1); eq (10-6.4)
        for i in range(self.n_gas):
            denom = 0.
            for j in range(self.n_gas):
                ϕ = (1 + (η_i[i]/η_i[j])**0.5*(self.X_μ[j]/self.X_μ[i])**0.25)**2/(4./np.sqrt(2)*(1 + self.X_μ[i]/self.X_μ[j])**0.5)
                denom += self.X[j]*ϕ
            K += self.X[i]*K_i[i]/denom
        return K

    def calc_c_p(self,T):
        '''
        calculate specific heat capacity at constant pressure for given T
        source: A.6 of Fundamentals of Thermodynamics Borgnakke & Sonntag (2009)
        c_p_i(T) = 1000*(sum over j of c_p_j*T^j/1000) for j=0-3
        c_p(T) = sum over i of c_p_i(T)*q_i

        inputs:
            * self
            * T [K] - local temperature
        outputs:
            * c_p [J/kg/K] - specific heat capacity
        '''
        θ = T/1000.
        c_p = 1./self.μ_air*np.sum(self.X_μ*self.X*(self.X_cp[:,0] + self.X_cp[:,1]*θ + self.X_cp[:,2]*θ**2 + self.X_cp[:,3]*θ**3))
        return c_p*1000

    def calc_η(self):
        '''
        calculate dynamic viscosity of air (η) as a function of local pressure
        Reid+ (1977) eq (9-3.9) (pg 395)
        follow Wilkes' rule for mixtures for combining different gases into "air"
        Reid+ (1977) eq (9-5.2) (pg 411)

        input:
            * self
        output:
            * η [Pa s] - dynamic viscosity
        '''
        η_i = 1e-7*26.69*(self.X_μ*1e3*self.T)**0.5/self.X_d**2/self.calc_Ω_η(self.T/self.X_ep_kB)
        η = 0.
        for i in range(self.n_gas):
            denom = 0.
            for j in range(self.n_gas):
                phi = (1 + (η_i[i]/η_i[j])**0.5*(self.X_μ[j]/self.X_μ[i])**0.25)**2/(4./np.sqrt(2)*(1 + self.X_μ[i]/self.X_μ[j])**0.5)
                denom += self.X[j]*phi
            η += self.X[i]*η_i[i]/denom
        return η

    def z2x4drdz(self,z):
        '''
        adjust altitude-sensitive values to a given z
        input:
            * self
            * z [m] - altitude
        '''
        self.z = z
        self.T = self.z2T(z)
        self.p = self.T2p(self.T)
        self.RH = self.pT2RH(self.p,self.T)
        self.p_c = self.f_c*self.p
        self.ρ = self.pT2ρ(self.p,self.T)
        self.η = self.calc_η()
        self.K = self.calc_K()
        self.D_c = self.calc_D_c()

    def calc_H(self,z):
        '''
        calculate scale height at given z
        inputs:
            * self
            * z [m] - height in atm for scale height calc
        output:
            * H [m] - scale height
        '''
        H = self.R_sp_air*self.z2T(z)/self.g
        return H

    def z2x4drdz_12(self,z):
        '''
        adjust altitude-sensitive values to a given z for 2 condensible system
        presently only works for Titan conditions, follows Graves et al. (2008)
        functions for atm properties
        input:
            * self
            * z [m] - altitude
        '''
        self.z = z
        self.T = self.z2T_Titan(z)
        self.p = self.z2p_Titan(z)
        self.p_c2 = self.z2fCH4_Titan(z)*self.p
        self.p_c1 = self.p - self.p_c2

        self.η = self.calc_ηG()
        self.K = self.calc_KG()
        self.D_c2 = self.calc_D_c2G()

        if self.is_Graves:
            # assume atm molar mass is pure N2
            μ = 28e-3 # [kg/mol]
        else:
            # self consistently calculate molar mass
            μ = np.sum(self.X_μ*np.array([self.p_c1/self.p,self.p_c2/self.p])) # [kg/mol]
        self.ρ = self.p/self.T/R_gas*μ # air density [kg/m3]
        self.D_c1 = self.calc_D_c1G()

    # use functions for atmospheric properties from Graves et al. (2008)
    def calc_ηG(self):
        '''
        dynamic viscosity of N2, from
        Graves et al. (2008) eq (A.11)
        input:
            * self
        output:
            * η [Pa s] - dynamic viscosity of N2 at local T
        '''
        return 1.718e-5+(5.1e-8*(self.T-273))
    def calc_KG(self):
        '''
        thermal conductivity of N2, from
        Graves et al. (2008) eq (A.35)
        input:
            * self
        output:
            * K [W/m/K] - thermal conductivity at local T
        '''
        return 9e-5*self.T+0.0005
    def calc_D_c2G(self):
        '''
        mass diffusion coefficient for CH4 in N2, from
        Graves et al. (2008) eq (A.17a)
        input:
            * self
        output:
            * D [m2/s] - diffusivity at local T, p
        '''
        T0 = 273.
        p0 = 1.01325e5
        return 1.96e-5*(self.T/T0)**1.75*(p0/self.p)
    def calc_D_c1G(self):
        '''
        mass diffusion coefficient for N2 in N2, from
        Graves et al. (2008) eq (A.18)
        input:
            * self
        output:
            * D [m2/s] - diffusivity at local T, p
        '''
        return 2.*self.η/self.ρ
