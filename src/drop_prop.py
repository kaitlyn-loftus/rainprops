import numpy as np
from scipy.optimize import brentq, newton
import matplotlib.pyplot as plt

R_gas = 8.31446261815324 # [J/K/mol]

#######################################################################
# functions associated with dimensionless numbers
#######################################################################

def calc_Re(r,v,pl):
    '''
    calculate Reynolds number
    Re = 2*ρ_fluid*r*v/η_fluid
    inputs:
        * r [m] - drop size
        * v [m/s] - drop fall speed
        * pl [Planet object]
    outputs:
        * Re [] - Reynolds number
    '''
    return 2*pl.ρ*r*np.abs(v)/pl.η

def calc_Cc(r,pl):
    '''
    calculate the Cunningham-Stokes correction factor
    dimensionless number to account for drag on small particles
    in between continuum and free molecular flow regimes

    (turn off calculating Cc as a property of Planet object)
    ultimately not used in LoWo21

    inputs:
        * r [m] - radius of the falling particle
        * pl [Planet] - Planet object with atmospheric properties

    output:
        * Cc [] - Cunningham-Stokes correction factor
    '''
    if pl.is_Cc:
        # mean free path of air
        # Seinfeld and Pandis eq 9.6 pg 399
        mfp = 2.*pl.η/(pl.p*(8.*pl.μ_air/(np.pi*R_gas*pl.T))**0.5) # [m]
        # Knudsen number
        Kn = mfp/r # []
        # Cunningham-Stokes correction factor
        # assuming drop is liquid
        # from Allen & Rabe (1982) eq (24)
        Cc = 1 + Kn*(1.55+0.471*np.exp(-0.596/Kn)) #[]
    else:
        Cc = 1.
    return Cc

# calculate different variations of Weber number

def calc_We(r,pl):
    '''
    calculate We number
    We = r*v^2*ρ_air/σ
    inputs:
        * r [m] - equiv r
        * pl [Planet object]
    output:
        * We [] - Weber number
    '''
    We = r*calc_v(r,pl,pl.f_rat,pl.f_C_D)**2*pl.ρ/pl.c.σ(pl.T)
    return We

def calc_We_spherical(r,pl):
    '''
    calculate We number assuming spherical raindrop
    We = r*v^2*ρ_air/σ
    inputs:
        * r [m] - equiv r
        * pl [Planet object]
    output:
        * We [] - Weber number
    '''
    v = calc_v(r,pl,pl.f_rat,calc_C_D_sphere)
    We = r*v**2*pl.ρ/pl.c.σ(pl.T)
    return We

def calc_We_CD1(r,pl):
    '''
    calculate We number assuming spherical raindrop and C_D = 1
    We = r*v^2*ρ_air/σ
    inputs:
        * r [m] - equiv r
        * pl [Planet object]
    output:
        * We [] - Weber number
    '''
    v = np.sqrt(8./3*r*pl.g*(pl.c.ρ(pl.T))/pl.ρ)
    We = r*v**2*pl.ρ/pl.c.σ(pl.T)
    return We


def δ0(δ,pl,f):
    '''
    zero function for numerical solver
    to calculate temperature difference between air and drop
    inputs:
        * δ [K] - T_air - T_drop
        * pl [Planet object]
        * f [] - ventilation factor
    output:
        * difference between δ predicted by given δ, pl and given δ [K]
    '''
    ΔρcΔT = (pl.c.p_sat(pl.T)/pl.T-pl.c.p_sat(pl.T-δ)/(pl.T-δ))/δ/pl.c.R
    A = (pl.K*f[1]+pl.c.L(pl.T)*pl.D_c*ΔρcΔT*f[0])
    B = pl.D_c*pl.c.L(pl.T)*f[0]*((1-pl.RH)*pl.c.p_sat(pl.T)/pl.c.R/pl.T)
    return δ - B/A

def calc_drdz_Λ(r,pl,z,w=0):
    '''
    calculate drdz for evaluating Λ, LoWo eq ()
    inputs:
        * r [m] - equiv r
        * pl [Planet object]
        * z [m] - altitude for setting z-dependent atm properties
        * (optional) w [m/s] - vertical wind speed
    '''
    pl.z2x4drdz(z)
    v = w-1*calc_v(r,pl,pl.f_rat,pl.f_C_D)
    f = pl.f_vent(r,v,pl)
    # solve for temperature difference between T_air and T_drop (δ)
    if pl.T-pl.T_LCL<=0.:
        δ = 0.
    else:
        δ = brentq(δ0,1e-6,pl.T-pl.T_LCL,args=(pl,f))

    T_drop = pl.T - δ

    dmdt = 4.*np.pi*r*pl.D_c*f[0]*(pl.p_c/pl.c.R/pl.T - pl.c.p_sat(T_drop)/pl.c.R/T_drop)
    drdt = dmdt/(4.*np.pi*r**2*pl.c.ρ(T_drop))
    drdz = drdt*1./v
    return drdz

def calc_Λ(r,pl,ell,w=0,z=None):
    '''
    calculate Λ LoWo eq()
    inputs:
        * r [m] - equiv r
        * pl [Planet object]
        * ell [m] - length scale overwhich to consider falling + evaporation
        * (optional) w [m/s] - vertical wind speed
        * (optional) z [m] - specify altitude where z-dependent values should be evaluated,
                             default is None which then assumes z = z_LCL - ℓ/2
    output:
        * Λ []
    '''
    if z==None:
        z = pl.z_LCL - ell/2.
    drdz = calc_drdz_Λ(r,pl,z,w)
    Λ = drdz*3*ell/r
    return Λ

def r_Λ_x(r,pl,ell,Λ_val,w):
    '''
    zero function for numerical solver
    to calculate r for given Λ_val, ℓ, pl
    inputs:
        * r [m] - equiv r
        * pl [Planet object]
        * ell [m] - length scale overwhich to consider falling + evaporation
        * Λ_val [] - Λ value solving for
        * w [m/s] - vertical wind speed
    output:
        * difference between Λ predicted by given r, ℓ and desired Λ []
    '''
    Λ = calc_Λ(r,pl,ell)
    return Λ_val - Λ

def calc_r_from_Λ(pl,ell,Λ_val=1,w=0):
    '''
    zero function for numerical solver
    to calculate r for given Λ_val, ℓ, pl
    inputs:
        * r [m] - equiv r
        * pl [Planet object]
        * ell [m] - length scale overwhich to consider falling + evaporation
        * Λ_val [] - Λ value solving for
        * (optional) w [m/s] - vertical wind speed
    output:
        * r [m] - r_eq such that Λ(r,ℓ) = Λ_val
    '''
    return brentq(r_Λ_x,1e-7,1e-3,args=(pl,ell,Λ_val,w))

#######################################################################
# functions associated with raindrop shape calculations
#######################################################################
def req_0(rat,req,pl):
    '''
    zero function for numerical solver
    to calculate (equilibrium) shape of drop as an oblate spheriod
    inputs:
        * rat [] - ratio of spheriod b/a
        * req [m] - equivalent volume spherical radius
        * pl [Planet object]
    output:
        * difference between drop size predicted by given input ratio and actual drop size [m]
    '''
    return req-(pl.c.σ(pl.T)/pl.g/(pl.c.ρ(pl.T)-pl.ρ))**(0.5)*rat**(-1./6)*(rat**(-2)-2*rat**(-1./3)+1)**0.5

def calc_rat_Green(req,pl,v=None):
    '''
    calculate (equilibrium) shape of drop as an oblate spheriod
    see Green (1975) eq (6) [note ∃ a small typo in G75, σ should be raised to 0.5 rather than 1]
    also LoWo eq (2)
    inputs:
        * req [m] - equivalent volume spherical radius
        * pl [Planet object]
        * (optional) v [m/s] - velocity, not used but taken to have same arguments required as Lorenz (1993) shape calculation
    output:
        * rat [] - axis ratio b/a
    '''
    return brentq(req_0,1e-2,1,args=(req,pl))

def a_0(rat,a,pl):
    '''
    zero function for numerical solver
    to calculate (equilibrium) shape of drop as an oblate spheriod
    with knowledge of semimajor axis
    inputs:
        * rat [] - ratio of spheroid b/a
        * a [m] - semimajor axis of spheroid
        * pl [Planet object]
    output:
        * difference between drop semimajor axis predicted by given input ratio and given semimajor axis [m]
    '''
    return a-rat**(-1./3)*(pl.c.σ(pl.T)/pl.g/(pl.c.ρ(pl.T) - pl.ρ))**(0.5)*rat**(-1./6)*(rat**(-2)-2*rat**(-1./3)+1)**0.5

def calc_rat_Green_from_a(a,pl):
    '''
    calculate (equilibrium) shape of drop as an oblate spheriod
    from known semimajor axis (rather than r_eq)
    use a = rat^(-1/3)r_eq with:
        Green (1975) eq (6) [note ∃ a small typo in G75, σ should be raised to 0.5 rather than 1]
        or LoWo eq (2)
    inputs:
        * a[m] - semimajor axis of drop
        * pl [Planet object]
    output:
        * rat [] - axis ratio b/a

    '''
    return brentq(a_0,0.01,1,args=(a,pl))

def calc_rat_Lorenz(r,pl,v):
    '''
    Lorenz (1993)'s parametrization of b/a (rat) by Weber #
    linear fit to k, so can become negative (which does not make sense physically)
    therefore arbitrarily set b/a < 0 to an arbitrarily small positive number
    inputs:
        * req [m] - equivalent volume spherical radius
        * pl [Planet object]
        * v [m/s] - velocity
    output:
        * rat [] - axis ratio b/a

    '''
    We = r*v**2*pl.ρ/pl.c.σ(pl.T)
    if We>=0.1: # L93 eq (11)
        k = 0.97 - 0.072*We
    else: # L93 eq (12)
        k = (1 - (9.*We/16))**0.5
    if k<=0.1: # KL correction to prevent negative values
        k = 1e-1
    return k

def calc_Ψ(r,pl):
    '''
    calculate sphericity = Ψ = surface area of sphere of eq volume / actual surface area
    inputs:
        * r [m] - radius of drop
        * atm [Atm object] - describes atmosphere drop falling in
    output:
        * Ψ [] - surface area of sphere of eq volume / actual surface area
    '''
    return calc_SA_sphere(r)/calc_SA_spheriod(r,pl)


calc_SA_sphere = lambda r: 4*np.pi*r**2 # [m2] surface area of a sphere of radius r
delta = lambda a0,a: (a/a0)**2 -1 # defined in Green (1975)

def calc_SA_spheriod(r,pl):
    '''
    calculate surface area of a spheriod
    inputs:
        * r [m] - radius of drop
        * atm [Atm object] - describes atmosphere drop falling in
    output:
        * surface area of spheriod [m2]
    '''
    rat = calc_rat_Green(r,pl)
    a = r/rat**(1./3)
    d = delta(r,a)
    eps = np.sqrt(1 - (1+d)**(-3))
    return 2*np.pi*a**2*(1+rat**2*np.arctanh(eps)/eps)

#######################################################################
# functions to assign more functions for Planet objects
#######################################################################

def assign_f_rat(rat_param):
    '''
    input:
        * rat_param [string] - parametrization for axis ratio calculation
    output:
        * f_rat [function]- function to calculate axis ratio for given parametrization
    '''
    if rat_param == 'G':
        f_rat = calc_rat_Green
    elif rat_param == 'L':
        f_rat = calc_rat_Lorenz
    else:
        raise Exception('invalid input for "rat_param"')
    return f_rat

def assign_f_C_D(C_D_param):
    '''
    input:
        * C_D_param [string] - parameterization for drag coefficient C_D
    output:
        * f_C_D [function] - function to calculate C_D for given parametrization
    '''
    if C_D_param == 'K8':
        f_C_D = calc_C_D_Loth
    elif C_D_param == 'L':
        f_C_D = calc_C_D_Lorenz
    elif C_D_param == 'S':
        f_C_D = calc_C_D_Salman
    elif C_D_param == 'H':
        f_C_D = calc_C_D_Holzer
    elif C_D_param == 'G_N':
        f_C_D = calc_C_D_Ganser_Newton
    elif C_D_param == 'sphere':
        f_C_D = calc_C_D_sphere
    else:
        raise Exception('invalid input for "C_D_param"')
    return f_C_D

def assign_f_vent(f_vent_param):
    '''
    input:
        * f_vent_param [string] - parameterization for ventilation coefficients
                                  for molecular and heat transport
    output:
        * f_C_D [function] - function to calculate ventilation coefficients
    '''
    # assume f_V_mol = f_V_heat
    if f_vent_param=='m':
        f_f_vent = calc_f_vent_mass
    # calculate molecular transport and heat transport separately
    elif f_vent_param=='mh':
        f_f_vent = calc_f_vent_mass_heat
    # assume f_V_mol = f_V_heat = 1
    elif f_vent_param=='1':
        f_f_vent = calc_f_vent_as_1
    else:
        raise Exception('invalid input for "f_vent_param"')
    return f_f_vent

#######################################################################
# drag coefficient (C_D) functions
# all take same input parameters for easy exchangabilty
#######################################################################
def calc_C_D_Loth(r,v,pl):
    # C_D parameterization given in LoWo, eqs ()
    Re = calc_Re(r,v,pl)
    try:
        rat = calc_rat_Green(r,pl,v)
    except:
        rat = 1.
    if rat<0.999:
        try:
            eps = (1 - rat**2)**0.5
            A_star_surf = rat**(-2./3)/2. + rat**(4./3)/4./eps*np.log((1 + eps)/(1 - eps)) # Loth et al. (2008) eq (21a)
            C = 1 + 1.5*(A_star_surf-1)**0.5 + 6.7*(A_star_surf-1) # Loth et al. (2008) eq (25)
        except:
            C =1
    else:
        C = 1.
    C_D =  24./Re*(1+0.15*Re**(229./333))+0.42/(1 + 4.25e4*Re**(-1.16))
    C_D = C_D*C
    return C_D

C_D_Lorenz_ = lambda Re: (24./Re)*(1 + 0.197*Re**0.63 + 2.6e-4*Re**1.38)
C_D_C6_ = lambda Re: 24./Re*(1+0.15*Re**0.687)+0.42/(1 + 4.25e4*Re**(-1.16))

def calc_C_D_Lorenz(r,v,pl):
    '''
    calculate drag coefficient of a spheriod
    parametrizes C_D from that of a sphere of eq volume linearly
    source: Lorenz (1993)
    '''
    Re = calc_Re(r,v,pl)
    rat = calc_rat_Lorenz(r,pl,v)
    C_D = C_D_Lorenz_(Re)/rat
    return C_D


def calc_C_D_sphere(r,v,pl):
    Re = calc_Re(r,v,pl)
    C_D = C_D_C6_(Re)
    return C_D

def calc_C_D_Holzer(r,v,atm):
    Re = calc_Re(r,v,atm)
    rat = calc_rat_Green(r,atm,v)
    a = rat**(-1./3)*r
    ϕ_perp = r**2/a**2
    ϕ = calc_SA_sphere(r)/calc_SA_spheriod(r,atm)
    C_D = 8./Re/ϕ_perp**(0.5) + 16./Re/ϕ**(0.5) + 3./Re**0.5/ϕ**(0.75) +0.4210**(0.4*((-1*np.log10(ϕ))**0.2))/ϕ_perp
    return C_D

def calc_C_D_Ganser_Newton(r,v,atm):
    ϕ = calc_SA_sphere(r)/calc_SA_spheriod(r,atm)
    C_D = 0.42*10**(1.8148*(-np.log10(ϕ))**0.5743)
    return C_D

# parameterized C_D
# Salman & Verba 1986
a_ = lambda Ψ: 794.889*Ψ**4 - 2294.985*Ψ**3 + 2400.77*Ψ**2 - 1090.0719*Ψ + 211.686
b_ = lambda Ψ: -320.757*Ψ**4 + 933.336*Ψ**3 - 973.461*Ψ**2 + 433.488*Ψ - 67
c_ = lambda Ψ: (22.265*Ψ**4 - 35.241*Ψ**3 + 20.365*Ψ**2 - 4.131*Ψ + 0.304)**(-1)
C_D_ = lambda Re, Ψ: a_(Ψ)/Re + b_(Ψ)/np.sqrt(Re) + c_(Ψ)

def calc_C_D_Salman(r,v,pl):
    '''
    calculate drag coefficient of a spheriod
    parameterizes C_D as a function of deviation of surface area from that of a
    sphere and Re of sphere of eq volume
    source: Salman & Verba (1986)
    inputs:
        * r [m] - radius of drop
        * v [m/s] - velocity of drop
        * pl [Planet object] - describes atmosphere drop falling in
    output:
        * C_D [] - drag coefficient of drop
    '''
    Re = calc_Re(r,v,pl)
    Ψ = calc_SA_sphere(r)/calc_SA_spheriod(r,pl)
    return C_D_(Re,Ψ)

def calc_C_D_after(r,v,pl,f_rat=calc_rat_Green):
    '''
    calculate C_D from knowledge of r, v, and pl
    inputs:
        * r [m] - equiv r
        * v [m/s] - raindrop velocity
        * pl [Planet object]
        * (optional) f_rat [function]
    '''
    rat = f_rat(r,pl,v)
    Re = calc_Re(r,v,pl)
    C_D = 8./3*pl.c.ρ(pl.T)/pl.ρ*r*rat**(2./3)*pl.g/v**2
    return C_D, rat, Re

#######################################################################
# velocity functions
#######################################################################

def v_0(v,r,pl,f_rat,f_C_D):
    '''
    zero function for numerical solver
    to calculate terminal velocity
    inputs:
        * v [m/s] - guess terminal velocity
        * r [m] - raindrop equiv radius
        * pl [Planet object]
        * f_rat [function] - function to calculate raindrop axis ratio
        * f_C_D [function] - function to calculate drag coefficient C_D
    output:
        * difference between v predicted by given v and given v [m/s]
    '''
    try:
        rat = f_rat(r,pl,v) # []
    except:
        rat = 1. # fails sometimes when rat is very close to 1
    A = r**2*rat**(-2./3) # [m2] cross sectional area of oblate spheroid
    C_D = f_C_D(r,v,pl) # []
    if r<1e-9: # avoid errors from ode solver trying unphysical values
        return 0.
    else:
        return v - np.sqrt(8./3*r**3/A*pl.g*(pl.c.ρ(pl.T)-pl.ρ)/pl.ρ/C_D*calc_Cc(r,pl))

def calc_v(r,pl,f_rat,f_C_D,v_min=1e-10,v_max=300):
    '''
    calculate raindrop terminal velocity numerically

    inputs:
        * r [m] - raindrop equivalent radius
        * pl [Planet object]
        * f_rat [function] - function to calculate raindrop axis ratio
        * f_C_D [function] - function to calculate drag coefficient C_D
        * (optional) v_min [m/s] - minimum bound on v to solve for
        * (optional) v_max [m/s] - maximum bound on v to solve for
        {Note: in extreme planetary environments v_min and v_max may need to be fiddled with for solver to work}
    output:
        * v [m/s] - terminal velocity
    '''
    try:
        # first try using Newton root solver
        v = newton(v_0,1.5, args=(r,pl,f_rat,f_C_D),maxiter=100)
    except:
        # then try a bounded root solver if it fails
        v = brentq(v_0,1e-3,10.,args=(r,pl,f_rat,f_C_D))
    return v


def calc_v_Earth_Beard(r,pl,is_CC_correction=True):
    '''
    calculate terminal velocity of raindrop of equivalent radius r
    at various heights in Earth's atmosphere
    input:
        * r [m] - equivalent raindrop radius
        * pl [Planet object]
        * (optional) is_CC_correction - whether or not to use Cunningham-Stokes correction, default is True
    output:
        * v [m/s] - terminal velocity
    follows Beard 1976 semi-empirical raindrop velocities
    his Table 1 summarizes eqs involved
    uses cgs units, converted here to SI

    Beard breaks raindrops down into three size-based categories
    and parameterizes v_terminal based on a mix of dimensional
    analysis and fitting observational data
    '''
    d = 2*r # [m] equivalent diameter of raindrop
    # Beard works in d instead of r
    delta_ρ = pl.c.ρ(pl.T) - pl.ρ # [kg/m3] difference in density between drop and air

    if is_CC_correction:
        # Cunningham-Stokes correction factor for slip drag
        mfp_0 = 6.62e-8 # [m]
        η_0 = 1.818e-5 # [kg/m/s]
        p_0 = 1.01325e5 # [Pa]
        T_0 = 293.15 # [K]
        mfp = mfp_0*(pl.η/η_0)*(p_0/pl.p)*(pl.T/T_0)**0.5
        C_CS = 1 + 2.51*mfp/d
    else:
        C_CS = 1.

    if d < 19e-6:
        C_D = delta_ρ*pl.g/18./pl.η
        v = C_D*C_CS*d**2
    elif d < 1.07e-3:
        C_D = 4./3*pl.ρ*delta_ρ*pl.g/pl.η**2 # [] drag coefficient
        N_Da = C_D*d**3 # [] Davies number
        b = np.array([-0.318657e1,0.992696,-0.153193e-2,-0.987059e-3,-0.578878e-3,0.855176e-4,-0.327815e-5]) # fit coefficients
        X = np.log(N_Da)
        Y = 0
        for i in range(7):
            Y += b[i]*X**i
        N_Re = C_CS*np.exp(Y)
        v = pl.η*N_Re/pl.ρ/d
    else:
        N_P = pl.c.σ(pl.T)**3*pl.ρ**2/pl.η**4/delta_ρ/pl.g # [] number
        C_D = 4/3.*delta_ρ*pl.g/pl.c.σ(pl.T) # [] drag coefficient
        b = np.array([-0.500015e1,0.523778e1,-0.204914e1,0.475294,-0.542819e-1,0.238449e-2]) # fit coefficients
        Bo = C_D*d**2 # [] Bond number
        X = np.log(Bo*N_P**(1./6))
        Y = 0
        for i in range(6):
            Y += b[i]*X**i
        N_Re = N_P**(1./6)*np.exp(Y)
        v = pl.η*N_Re/pl.ρ/d
    return v

################################################################
# functions to calculate ventilation coefficients
################################################################

def calc_f_vent_as_1(r,v,pl):
    '''
    '''
    return np.array([1.,1.])

def calc_f_vent_mass(r,v,pl):
    '''
    calculate evaporation enhancement factor from ventilation
    due to falling
    assume f_vents for molecules and heat are the same
    source: Pruppacher & Klett (2010) pg 443
    LoWo eq ()
    inputs:
        * r [m] - equiv radius
        * v [m/s] - velocity relative to air
        * pl [Planet object]
    output:
        * f_vent_mol []
        * f_vent_heat [] - assumed to be the same as f_vent_mol in this function
    '''
    Re = calc_Re(r,v,pl)
    Sc = pl.η/pl.D_c/pl.ρ
    X = Re**0.5*Sc**(1./3)
    if X<1.4:
        f = 1 + 0.108*X**2
    else:
        f = 0.78 + 0.308*X
    return np.array([f,f])

def calc_f_vent_mass_heat(r,v,pl):
    '''
    calculate evaporation enhancement factor from ventilation
    due to falling
    f_vents for molecules and heat are calculated differently
    source: Pruppacher & Klett (2010) pg 443
    LoWo eq ()
    inputs:
        * r [m] - equiv radius
        * v [m/s] - velocity relative to air
        * pl [Planet object]
    output:
        * f_vent_mol []
        * f_vent_heat [] - assumed to be the same as f_vent_mol in this function
    '''
    Re = calc_Re(r,v,pl)

    Sc = pl.η/pl.D_c/pl.ρ # [] Schmidt number
    Pr = pl.η/pl.K*pl.c_p # [] Prandtl number

    X = Re**0.5*np.array([Sc,Pr])**(1./3)
    f = np.where(X<1.4,1 + 0.108*X**2,0.78 + 0.308*X)
    return f


################################################################
# functions to calculate maximum stable raindrop
################################################################

def calc_r_max_CL17(pl):
    '''
    maximum stable raindrop size as (implicitly) calculated by Craddock & Lorenz (2017)
    input:
        * pl [Planet object] - describes planet environment
    output:
        * r_max [m^2.5/kg^0.5] - maximum raindrop radius before breakup
    '''
    r_max = np.sqrt(3./2*pl.c.σ_const/pl.g/pl.ρ/pl.c.ρ_const) # [m^2.5/kg^0.5]
    return r_max

def calc_r_max_fb_req(pl):
    '''
    maximum stable raindrop from force balance (fb)
    ℓ_max = 2π*req
    inputs:
        * pl [Planet object] - describes planet environment
    output:
        * r_max [m] - maximum stable raindrop size
    '''
    r_max = np.sqrt(3./2*pl.c.σ(pl.T)/pl.g/(pl.c.ρ(pl.T)-pl.ρ))
    return r_max

def calc_r_max_fb_a_0(r_max,pl):
    '''
    zero function for numerical solver
    to calculate maximum stable raindrop size from force balance (fb)
    ℓ_max = 2π*a
    inputs:
        * r_max [m] - equiv r
        * pl [Planet object] - describes planet environment
    output:
        * difference between r_max predicted by given r and given r [m]
    '''
    rat = calc_rat_Green(r_max,pl)
    return r_max - (3./2*pl.c.σ(pl.T)/pl.g/(pl.c.ρ(pl.T)-pl.ρ)*rat**(-1./3))**0.5

def calc_r_max_fb_a(pl):
    '''
    maximum stable raindrop from force balance (fb)
    ℓ_max = 2π*a
    inputs:
        * pl [Planet object] - describes planet environment
    output:
        * r_max [m] - maximum stable raindrop size
    '''
    return brentq(calc_r_max_fb_a_0,1e-5,50e-3,args=(pl))

def calc_r_max_RT(pl,ℓ_div_a):
    '''
    maximum stable raindrop from Rayleigh Taylor instablity
    ℓ_max = ℓ_div_a*a
    inputs:
        * pl [Planet object]
        * ℓ_div_a [] - factor times a to give ℓ_max
    output:
        * r_max [m] - maximum stable raindrop equivalent radius
    '''
    a_max = 1./ℓ_div_a*np.pi*np.sqrt(pl.c.σ(pl.T)/pl.g/(pl.c.ρ(pl.T) - pl.ρ)) # [m]
    rat = calc_rat_Green_from_a(a_max,pl) # [] numerically calculate b/a, knowing a
    r_max = a_max*rat**(1./3) # [m] from Green, b/a = (r_eq/a)^3
    return r_max

def calc_r_max_RT_r(pl,ℓ_div_r):
    '''
    maximum stable raindrop from Rayleigh Taylor instablity
    ℓ_max = ℓ_div_r*r
    inputs:
        * pl [Planet object]
        * ℓ_div_r [] - factor times r to give ℓ_max
    output:
        * r_max [m] - maximum stable raindrop equivalent radius
    '''
    return 1./ℓ_div_r*np.pi*np.sqrt(pl.c.σ(pl.T)/pl.g/(pl.c.ρ(pl.T) - pl.ρ)) # [m]


def calc_r_max_L93_0(r,pl):
    '''
    zero function for numerical solver
    to calculate maximum stable raindrop size following Lorenz (1993)
    inputs:
        * r [m] - equiv r
        * pl [Planet object] - describes planet environment
    output:
        * difference between We predicted by given r and critical We = 4 [ ]
    '''
    return r*calc_v(r,pl,pl.f_rat,pl.f_C_D)**2*pl.ρ/pl.c.σ(pl.T) - 4

def calc_r_max_L93(pl):
    '''
    calculate maximum stable raindrop size following Lorenz (1993)
    We = 4, v is calculated numerically
    input:
        * pl [Planet object] - describes planet environment
    output:
        * r_max [m] - maximum stable raindrop size
    '''
    r_max = brentq(calc_r_max_L93_0,1e-6,calc_r_max_fb_req(pl)*10,args=(pl))
    return r_max

def calc_r_max_P20(pl):
    '''
    calculate maximum stable raindrop size following Palumbo et al. (2020)
    We = 4, C_D = 1, b/a = 1
    input:
        * pl [Planet object] - describes planet environment
    output:
        * r_max [m] - maximum stable raindrop size
    '''
    return (3./2*pl.c.σ(pl.T)/pl.g/pl.c.ρ(pl.T))**0.5
