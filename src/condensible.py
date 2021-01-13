################################################################
# class for storing and calculating condensible attributes
################################################################

import numpy as np

# load condensible data
c_properties = np.genfromtxt('./data/condensible_properties.csv',delimiter=',',names=True)

# N2 + CH4 mixture activity coefficients, used in calc_γ
# from Thompson et al. (1992) eq (7b) and Table 1
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


class Condensible:
    '''
    Condensible class
    condensible properties
    attributes:
        * type_c [string] - molecular formula of condensible
                            currently limited to h2o, ch4, n2, or combo
                            (set up should be straightforward to add
                            other liquid condensibles if data
                            available and interest)
        * μ [kg/mol] - condensible molar mass
        * R [J/kg/K] - condensible specific gas constant
        * ρ_const [kg/m3] - constant condensible liquid density
        * L_const [J/kg] - constant latent heat of vaporization
        * T_crit [K] - critical T (where L=0)
        * T_freeze [K] - T at which condensible becomes solid
        * c_p [J/kg/K] - specific heat capacity of condensible liquid
        * conds_i [int] - index within data file
    methods:
        * __init__ - constructor
        methods which return T-dependent condensible properties:
            * σ [J/m2] - surface tension
            * p_sat [Pa] - condensible saturation pressure with T
            * L [J/kg] - T-dependent latent heat of vaporization
            * ρ [kg/m3] - T-dependent liquid condensible density
    '''
    def __init__(self,type_c):
        '''
        constructor for Condensible class
        initialize Condensible object
        inputs:
            * type_c [string] - molecular formula of condensible
        '''
        self.type_c = type_c
        if type_c!='h2o' and type_c!='ch4' and type_c!='n2' and type_c!='combo':
            raise Exception('this condensible species is not supported. use h2o, ch4, n2, or combo.')
        if self.type_c=='h2o':
            self.conds_i = 0
            # set constants for p_sat calculation
            # from Wagner & Pruß (2002) eq (2.5)
            self.T_c = 647.096
            self.p_c = 22.064*1e6
            self.a1 = -7.85951783
            self.a2 = 1.84408259
            self.a3 = -11.7866497
            self.a4 = 22.6807411
            self.a5 = -15.961879
            self.a6 = 1.80122502
        elif self.type_c=='ch4':
            self.conds_i = 1
            self.is_lorenz = False
        elif self.type_c=='n2':
            self.conds_i = 3
        elif self.type_c=='fe':
            self.conds_i = 2
        if type_c!='combo':
            self.μ = c_properties['molar_mass'][self.conds_i] # [kg/mol]
            self.R = c_properties['R_sp'][self.conds_i] # [J/kg/K], specific gas constant
            self.ρ_const = c_properties['rho_const'][self.conds_i] # [kg/m3]
            # see Table 3 for citations
            self.gas_i = int(c_properties['gas_prop_index'][self.conds_i]) # [int]
            self.L_const = c_properties['L_const'][self.conds_i] # [J/kg]
            # see Table 3 for citations
            self.T_crit = c_properties['T_crit'][self.conds_i] # [K]
            self.T_freeze = c_properties['T_freeze'][self.conds_i] # [K]
            # see Table 3 for citations
            self.σ_const = c_properties['gamma_const'][self.conds_i] # [N/m]
            # see Table 3 for citations
            self.c_p = c_properties['c_p'][self.conds_i] # [J/kg/K] for liquid condensible, ignore T-dependence (which is very limited)
            # cp H2O and N2 from CRC handbook of chemistry and physics ch 6
            # cp CH4 from Colwell (1964)

    def p_sat(self,T):
        '''
        calculate saturation partial pressure of condensible for a given temperature

        input:
            * T [K] - local temperature

        output:
            * p [Pa] - saturation partial pressure of condensible
        '''
        if self.type_c=='h2o':
            # from Wagner & Pruß (2002); eq (2.5)
            θ = 1 - T/self.T_c
            return self.p_c*np.exp(self.T_c/T*(self.a1*θ + self.a2*θ**1.5 + self.a3*θ**3 + self.a4*θ**3.5 + self.a5*θ**4 + self.a6*θ**7.5))
        elif self.type_c=='ch4':
            # from Graves+ (2008); appendix eq (A.27)
            return 3.4543e9*np.exp(-1145.705/T) # [Pa]
        elif self.type_c=='n2':
            # from Graves (2008); appendix eq (A.26)
            return 1e5*(10**(3.95-306./T)) # [Pa]


    def L(self,T):
        '''
        calculate latent heat of vaporization

        input:
            * T [K] - local temperature

        output:
            * Lv [J/kg] - water's latent heat of vaporization
        '''
        if self.type_c=='h2o':
            #linear fit to L data at below url, piecewise drop to 0 above critical T
            #https://www.engineeringtoolbox.com/water-properties-d_1573.html
            T = T - 273.15
            Lv = (-4.0227*T+2594.2)*1e3
            return np.where(T<=self.T_crit-273.15,Lv,0)
        elif self.type_c=='ch4':
            # from Graves et al. (2008) eq (A.36)
            return (10818-23.202*T)/self.μ
        elif self.type_c=='n2':
            # from Graves et al. (2008) eq (A.37)
            return (8679-40.702*T)/self.μ

    def ρ(self,T):
        '''
        calculate condensible liquid density as a function of temperature
        (presently h2o T dependence is handled at reference z only to
        ensure conservation of mass when integrating r)
        input:
            * self
            * T [K] - local temperature
        '''
        if self.type_c=='h2o':
            return self.ρ_const
        elif self.type_c=='ch4':
            if not(self.is_lorenz): # handle Lorenz (1993) assumption of liquid ρ
                # Graves et al. (2008) eq (A.5)
                return 612-1.8*T #[kg/m3]
            else:
                return self.ρ_const
        elif self.type_c=='fe':
            '''
            from Assael et al. (2006)
            good for 1809–2480 K
            '''
            c3 = 7034.96  #[kg/m3]
            c4 = 0.926  #[kg/m3/K]
            T_ref = 1811.0  #[K]
            ρ =  c3 - c4*(T-T_ref)
        elif self.type_c=='combo':
            return self.ρ_avg # assumed to be set externally
        else:
            return self.ρ_const # [kg/m3]

    def σ(self,T):
        '''
        calculate condensible liquid surface tension as a function of T
        input:
            * self
            * T [K] - local temperature
        output:
            * σ [N/m] - surface tension
        '''
        if self.type_c=='h2o':
            # from Vargaftik+ (1983)
            B = 235e-3 # [N/m]
            b = -0.625 # []
            mu = 1.256 # []
            Tc = 647.15 # [K]
            σ =  B*((Tc-T)/Tc)**mu*(1 + b*((Tc-T)/Tc))
            return σ
        elif self.type_c=='ch4':
            # return self.σ_const
            return 0.017 # [N/m] used by Lorenz (1993) and Graves et al. (2008)
        elif self.type_c=='fe':
            '''
            from Brillo & Egry (2005)
            linear fit to data
            '''
            σ_L = 1.92 #[N/m]
            σ_T = -3.97e-4 #[N/m/K]
            T_L = 1538 + 273.15 # [K]
            return σ_L + σ_T*(T - T_L)
        elif self.type_c=='combo':
            return self.σ_const

    ############################################

    def calc_γ(self,T,X1,X2):
        '''
        activity coefficient to account for thermodynamic
        effects of liquid condensible mixture
        presently only works for N2-CH4 mixture
        1 -> N2, 2 -> CH4
        from Thompson et al. (1992) eq (9)
        inputs:
            * self
            * T [K] - temperature
            * X1 [mol/mol] - molar concentration of component 1
            * X2 [mol/mol] - molar concentration of component 2
        output:
            * γ [ ] - condensible activity coefficient
        '''
        if self.type_c=='n2':
            a = calc_a(T)
            b = calc_b(T)
            c = calc_c(T)
            lnγ1 = X2**2*((a+3*b+5*c)-4*(b+4*c)*X2 + 12*c*X2**2)
            return np.exp(lnγ1)
        elif self.type_c=='ch4':
            a = calc_a(T)
            b = calc_b(T)
            c = calc_c(T)
            lnγ2 = X1**2*((a-3*b+5*c)+4*(b-4*c)*X1 + 12*c*X1**2)
            return np.exp(lnγ2)

    def calc_ρ_h2o(self,T):
        '''
        for h2o condensible, adjust ρ_const with a given T
        used to adjust ρ to account for T differences by assuming constant value from T_surf
        from Wagner & Pruß (2002) eq (2.6)
        inputs:
            * self
            * T [K]
        '''
        T_c = 647.096 # [K]
        θ = 1 - T/T_c
        ρ_c = 322. # [kg/m3]
        b1 = 1.99274064
        b2 = 1.09965342
        b3 = -0.510839303
        b4 = -1.75493479
        b5 = -45.5170352
        b6 = -6.74694450e5
        self.ρ_const = ρ_c*(1 + b1*θ**(1./3) + b2*θ**(2./3) + b3*θ**(5./3) + b4*θ**(16./3) + b5*θ**(43./3) + b6*θ**(110./3))
