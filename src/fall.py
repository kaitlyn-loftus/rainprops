################################################################
# module for liquid drop falling and evaporating in atmosphere
################################################################
import numpy as np
import scipy.integrate as integrate
import src.drop_prop as drop_prop
import functools
from scipy.optimize import brentq

##################################################
# DIFFERENTIAL EQs TO INTEGRATE
##################################################

def calc_drdz_dTdropdz(z,rT,pl,w=0):
    '''
    differential equations governing the evaporation of the falling rain drop
    dT_drop/dt is coupled rather than assuming equilibrium T_drop value
    inputs:
        * z [m] - altitude, integrating wrt z
        * rT [array] - integrated quantities
            - rT[0] = r [m] - equiv r
            - rT[1] = T_drop [K] raindrop temperature
        * pl [Planet object]
        * w [m/s] - vertical wind speed
    output:
        * drdz [m/m]
        * dTdz [K/m] - raindrop T change with z
    '''
    # integration should stop before here, but catch odd solver attempts
    if rT[0]<=1e-7:
        rT[0] = 1e-7
    if rT[1]>pl.T+5:
        rT[1] = pl.T+5
    elif rT[1]< 0.:
        rT[1] = pl.T_LCL

    # set planetary conditions to given altitude z
    pl.z2x4drdz(z)
    # calc dzdt
    v = w-1*drop_prop.calc_v(rT[0],pl,pl.f_rat,pl.f_C_D) # LoWo21 eq (9)
    if v>-1e-9: # integration should stop before here, but catch odd solver attempts
        v = -1e-9
    # calculate ventilation coefficients
    f = pl.f_vent(rT[0],v,pl)
    # calc drdt
    drdt = pl.D_c*f[0]/pl.c.R/rT[0]/pl.c.ρ(rT[1])*(pl.p_c/pl.T - pl.c.p_sat(rT[1])/rT[1]) # LoWo21 eq (10)
    # convert drdt to drdz using chain rule
    drdz = drdt*1./v # LoWo21 eq (15)
    # calc rate of change of drop T with time
    dTdt = 3./rT[0]/pl.c.c_p*(pl.c.L(rT[1])*drdt - pl.K*f[1]/rT[0]/pl.c.ρ(rT[1])*(rT[1] - pl.T)) # LoWo21 eq (11)
    # convert dTdt to dTdz via chain rule
    dTdz = dTdt*1./v
    # return appropriate derivatives
    return np.array([drdz,dTdz])


def δ0(δ,pl,f):
    '''
    zero function for numerical solver
    to calculate temperature difference between air and drop
    in equilibrium when drop evaporating
    LoWo21 eq (17)
    inputs:
        * δ [K] - T_air - T_drop
        * pl [Planet object]
        * f [ ] - ventilation factor
    output:
        * difference between δ predicted by given δ, pl and given δ [K]
    '''
    δ_pred = (pl.c.p_sat(pl.T-δ)/(pl.T-δ)-pl.RH*pl.c.p_sat(pl.T)/pl.T)
    δ_pred *= pl.D_c*f[0]*pl.c.L(pl.T)/pl.c.R/pl.K/f[1]
    return δ - δ_pred

def calc_δ_simple(pl,f):
    '''
    analytically estimate temperature difference between air and drop
    in equilibrium when drop evaporating
    LoWo21 eqs (17)-(18)
    inputs:
        * pl [Planet object]
        * f [ ] - ventilation factor
    output:
        * δ [K] - temperature difference between air and drop
                  in equilibrium when drop evaporating
    '''
    δ_guess =  (pl.T-pl.T_LCL)/2.
    δ = (pl.c.p_sat(pl.T-δ_guess)/(pl.T-δ_guess)-pl.RH*pl.c.p_sat(pl.T)/pl.T)
    δ *= pl.D_c*f[0]*pl.c.L(pl.T)/pl.c.R/pl.K/f[1]
    return δ


def calc_dtdz_drdz(z,tr,pl,w):
    '''
    differential equations governing the evaporation of the falling raindrop
    assume equilibrium T_drop value (solved numerically)
    inputs:
        * z [m] - altitude, integrating wrt z
        * tr [array] - integrated quantities
            - tr[0] = time [s] (t = 0 @ z = z_LCL)
            - tr[1] = r [m] - equiv r
        * pl [Planet object]
        * w [m/s] - vertical wind speed
    output:
        * dtdz [s/m]
        * drdz [m/m]
    '''
    if tr[1]<=1e-7: # integration should stop before here, but catch odd solver attempts
        tr[1] = 1e-7
    pl.z2x4drdz(z) # set planetary conditions to given altitude z
    v = w-1*drop_prop.calc_v(tr[1],pl,pl.f_rat,pl.f_C_D) # LoWo21 eq (9)
    if v>-1e-9:
        v = -1e-9
    f = pl.f_vent(tr[1],v,pl) # [ ] ventilation factor
    # calculate T_drop
    if pl.T-pl.T_LCL<=0.:
        δ = 0.
    else:
        δ = brentq(δ0,1e-6,pl.T-pl.T_LCL,args=(pl,f)) # LoWo21 eq (17)
        # δ = calc_δ_simple(pl,f)
    T_drop = pl.T - δ # [K]
    drdt = pl.D_c*f[0]/pl.c.ρ(T_drop)/tr[1]/pl.c.R*(pl.p_c/pl.T - pl.c.p_sat(T_drop)/T_drop) # LoWo21 eq (10)
    drdz = drdt*1./v # LoWo21 eq (15)
    # return appropriate derivatives
    return np.array([1./v,drdt*1./v])

def calc_dtdz_drdz_dTdz(z,trT,pl,w):
    '''
    differential equations governing the evaporation of the falling raindrop
    dT_drop/dt is coupled rather than assuming equilibrium T_drop value
    inputs:
        * z [m] - altitude, integrating wrt z
        * trT [array] - integrated quantities
            - trT[0] = t [s] - t = 0 @ z = z_LCL
            - trT[1] = r [m] - equiv r
            - trT[2] = T_drop [K] raindrop temperature
        * pl [Planet object]
        * w [m/s] - vertical wind speed
    output:
        * dtdz [s/m]
        * drdz [m/m]
        * dTdz [K/m] - raindrop T change with z
    '''
    if trT[1]<=1e-7: # integration should stop before here, but catch odd solver attempts
        trT[1] = 1e-7
    pl.z2x4drdz(z) # set planetary conditions to given altitude z

    v = w-1*drop_prop.calc_v(trT[1],pl,pl.f_rat,pl.f_C_D) # LoWo21 eq (9)
    if v>-1e-9: # integration should stop before here, but catch odd solver attempts
        v = -1e-9
    f = pl.f_vent(trT[1],v,pl) # [ ] ventilation factor
    drdt = pl.D_c*f[0]/pl.c.R/trT[1]/pl.c.ρ(trT[2])*(pl.p_c/pl.T - pl.c.p_sat(trT[2])/trT[2]) # LoWo21 eq (10)
    drdz = drdt*1./v # LoWo21 eq (15)
    dTdt = 3./trT[1]/pl.c.c_p*(pl.c.L(trT[2])*drdt - pl.K*f[1]/trT[1]/pl.c.ρ(trT[2])*(trT[2] - pl.T)) # LoWo21 eq (11)
    dTdz = dTdt*1./v # chain rule
    # return appropriate derivatives
    return np.array([1./v,drdz,dTdz])



def calc_drzvdt_w_a(t,rzvT,pl):
    '''
    differential equations governing the evaporation of the falling raindrop
    dT_drop/dt is coupled rather than assuming equilibrium T_drop value
    velocity is integrated rather than assuming terminal velocity
    inputs:
        * t [s] - time, integrating wrt t
        * rzvT [array] - integrated quantities
            - rzvT[0] = r [m] raindrop equiv radius
            - rzvT[1] = z [m] altitude
            - rzvT[2] = v [m/s] raindrop velocity
            - rzvT[3] = T_drop [K] raindrop temperature
        * pl [Planet object]
    output:
        * drdt [m/s]
        * dzdt [m/s]
        * dvdt [m/s^2]
        * dTdt [K/s]
    '''

    if rzvT[0]<=1e-7: # integration should stop before here, but catch odd solver attempts
        rzvT[0] = 1e-7
    pl.z2x4drdz(rzvT[1]) # set planetary conditions to given altitude z

    if rzvT[2]>-1e-9: # integration should stop before here, but catch odd solver attempts
        rzvT[2] = -1e-9

    if rzvT[3]>pl.T: # integration should stop before here, but catch odd solver attempts
        rzvT[3] = pl.T

    f = pl.f_vent(rzvT[0],rzvT[2],pl) # [ ] ventilation factor

    drdt = pl.D_c*f[0]/rzvT[0]/pl.c.ρ(rzvT[3])/pl.c.R*(pl.p_c/pl.T - pl.c.p_sat(rzvT[3])/rzvT[3]) # LoWo21 eq (10)
    dTdt = 3./rzvT[0]/pl.c.c_p*(pl.c.L(rzvT[3])*drdt - pl.K*f[1]/rzvT[0]/pl.c.ρ(rzvT[3])*(rzvT[3] - pl.T)) # LoWo21 eq (11)
    # calculate raindrop axis ratio [ ]
    try:
        rat = pl.f_rat(rzvT[0],pl,rzvT[2])
    except:
        rat = 1.
        print('rat failed')
    C_D = pl.f_C_D(rzvT[0],rzvT[2],pl) # [ ] calculate drag coefficient
    # calculate acceleration from Newton's 2nd law
    a = -pl.g - 3./8*rzvT[2]**2*rat**(-2./3)/rzvT[0]*pl.ρ/pl.c.ρ(rzvT[3])*C_D*np.sign(rzvT[2])/(drop_prop.calc_Cc(rzvT[0],pl)) # [m/s2]
    # return appropriate derivatives
    return np.array([drdt,rzvT[2],a,dTdt])

def calc_dzdt_w_a(t,zv,pl):
    '''
    differential equations governing the movement of a falling raindrop
    velocity is integrated rather than assuming terminal velocity
    no evaporation is considered as this eq is used for accessing terminal velocity assumption
    inputs:
        * t [s] - time, integrating wrt t
        * zv [array] - integrated quantities
            - zv[0] = z [m] altitude
            - zv[1] = v [m/s] raindrop velocity
        * pl [Planet object]
    output:
        * dzdt [m/s]
        * dvdt [m/s^2]
    '''
    pl.z2x4drdz(zv[0]) # set planetary conditions to given altitude z
    if zv[1]>-1e-9: # integration should stop before here, but catch odd solver attempts
        zv[1] = -1e-9
    # calculate raindrop axis ratio [ ]
    try:
        rat = pl.f_rat(pl.r,pl,zv[1])
    except:
        rat = 1.
    C_D = pl.f_C_D(pl.r,zv[1],pl) # [ ] calculate drag coefficient
    # calculate acceleration from Newton's 2nd law
    a = -pl.g - 3./8*zv[1]**2*rat**(-2./3)/pl.r*pl.ρ/pl.c.ρ(pl.T)*C_D/drop_prop.calc_Cc(pl.r,pl)*np.sign(zv[1]) # [m/s2]
    # return appropriate derivatives
    return np.array([zv[1],a])

def calc_dtdz_drdz_3pl(z,trT,pl,pl_F,pl_v,w):
    '''
    differential equations governing the evaporation of the falling raindrop
    dT_drop/dt is coupled rather than assuming equilibrium T_drop value
    input three different Planet objects to access different aspects of composition
    inputs:
        * z [m] - altitude, integrating wrt z
        * trT [array] - integrated quantities
            - trT[0] = t [s] - t = 0 @ z = z_LCL
            - trT[1] = r [m] - equiv r
            - trT[2] = T_drop [K] raindrop temperature
        * pl [Planet object] - governing atmospheric structure (H in LoWo21 Table 2)
        * pl_F [Planet object] - governing molecule and heat transport from drop (transport in LoWo21 Table 2)
        * pl_v [Planet object] - governing raindrop velocity (v_T in LoWo21 Table 2)
        * w [m/s] - vertical wind speed
    output:
        * dt/dz [s/m]
        * drdz [m/m]
        * dTdz [K/m] - raindrop T change with z

    '''
    if trT[1]<=1e-7: # integration should stop before here, but catch odd solver attempts
        trT[1] = 1e-7
    pl.z2x4drdz(z) # set planetary conditions to given altitude z
    # calc K and D_c using pl_F
    K = pl_F.calc_K(pl.T) # [W/K]
    D_c = pl_F.calc_D_c(pl.T,pl.p) # [m2/s]
    # calc v using pl_v
    v = w-1*drop_prop.calc_v(trT[1],pl_v,pl_v.f_rat,pl_v.f_C_D) # LoWo21 eq (9)
    if v>-1e-9: # integration should stop before here, but catch odd solver attempts
        v = -1e-9
    f = pl.f_vent(trT[1],v,pl_v) # [ ] ventilation factor, calc with pl_v
    # calc drdt and dTdt with pl
    drdt = D_c*f[0]/pl.c.R/trT[1]/pl.c.ρ(trT[2])*(pl.p_c/pl.T - pl.c.p_sat(trT[2])/trT[2]) # LoWo21 eq (10)
    drdz = drdt*1./v # LoWo21 eq (15)
    dTdt = 3./trT[1]/pl.c.c_p*(pl.c.L(trT[2])*drdt - K*f[1]/trT[1]/pl.c.ρ(trT[2])*(trT[2] - pl.T)) # LoWo21 eq (11)
    dTdz = dTdt*1./v # chain rule
    # return appropriate derivatives
    return np.array([1./v,drdt*1./v,dTdz])

# for Titan validation
# could be written better...
def calc_dtdz_dm12dt_dTdropdt(t,zm1m2T,pl,w=0):
    '''
    differential equations governing the evaporation of the falling raindrop
    dT_drop/dt is coupled rather than assuming equilibrium T_drop value
    inputs:
        * t [s] - altitude, integrating wrt t
        * zm1m2T [array] - integrated quantities
            zm1m2T[0] = z [m] altitude
            zm1m2T[1] = m1 [kg] raindrop mass of condensible 1
            zm1m2T[2] = m2 [kg] raindrop mass of condensible 2
            zm1m2T[3] = T_drop [K] raindrop temperature
        * pl [Planet object]
        * w [m/s] - vertical wind speed
    output:
        * dtdt [m/s] - v
        * dm1dt [kg/s] - raindrop condensible 1 mass change with z
        * dm2dt [kg/s] - raindrop condensible 2 mass change with z
        * dTdt [K/s] - raindrop T change with z
    '''
    # integration should stop before here, but catch odd solver attempts
    if zm1m2T[1]<0:
        zm1m2T[1] = 0.
    if zm1m2T[2]<0:
        zm1m2T[2] = 0.

    m = zm1m2T[1] + zm1m2T[2] # [kg] mass of drop
    pl.c.ρ_avg = (zm1m2T[1]/m/pl.c1.ρ(zm1m2T[3]) + zm1m2T[2]/m/pl.c2.ρ(zm1m2T[3]))**(-1) # [kg/m3] avg between components, Graves+ (2008) eq (A.7)
    r = (m/pl.c.ρ_avg*3./4/np.pi)**(1./3) # [m] equiv radius of drop
    X1 = zm1m2T[1]/pl.c1.μ/(zm1m2T[1]/pl.c1.μ + zm1m2T[2]/pl.c2.μ) # [mol/mol] drop liquid mole concentration of component 1
    # X1 + X2 = 1 =>
    X2 = 1. - X1 # [mol/mol], liquid mole concentration of component 2
    c_pℓ = zm1m2T[1]/m*pl.c1.c_p + zm1m2T[2]/m*pl.c2.c_p # [J/kg/K] avg between components

    if r<=1e-7: # integration should stop before here, but catch odd solver attempts
        r = 1e-7
    pl.z2x4drdz_12(zm1m2T[0]) # set planetary conditions to given altitude z

    # equivalent to LoWo21 eq (10)
    # follow Graves+ (2008) for accounting for 2 species
    if pl.is_Graves:
        # Graves+ (2008) eq (1)
        # difference from LoWo21 is Graves+ (2008) approxes conversion of ρ to p with an avg air-raindrop T instead of separate Ts
        Tf = (zm1m2T[3]+pl.T)/2. # [K]
        dm1dt = 4.*np.pi*r*pl.D_c1/Tf*(pl.p_c1/pl.c1.R - pl.c1.calc_γ(zm1m2T[3],X1,X2)*X1*pl.c1.p_sat(zm1m2T[3])/pl.c1.R)
        dm2dt = 4.*np.pi*r*pl.D_c2/Tf*(pl.p_c2/pl.c2.R - pl.c2.calc_γ(zm1m2T[3],X1,X2)*X2*pl.c2.p_sat(zm1m2T[3])/pl.c2.R)
    else:
        dm1dt = 4.*np.pi*r*pl.D_c1*(pl.p_c1/pl.c1.R/pl.T - pl.c1.calc_γ(zm1m2T[3],X1,X2)*X1*pl.c1.p_sat(zm1m2T[3])/pl.c1.R/zm1m2T[3])
        dm2dt = 4.*np.pi*r*pl.D_c2*(pl.p_c2/pl.c2.R/pl.T - pl.c2.calc_γ(zm1m2T[3],X1,X2)*X2*pl.c2.p_sat(zm1m2T[3])/pl.c2.R/zm1m2T[3])

    v = w-1*drop_prop.calc_v(r,pl,pl.f_rat,pl.f_C_D) # LoWo21 eq (9)
    if v>-1e-9: # integration should stop before here, but catch odd solver attempts
        v = -1e-9
    # calculate ventilation coefficients
    pl.D_c = pl.D_c1
    f1 = pl.f_vent(r,v,pl)
    pl.D_c = pl.D_c2
    f2 = pl.f_vent(r,v,pl)
    # account for f_vent_mol in dmdts
    dm1dt *= f1[0]
    dm2dt *= f2[0]
    # equivalent to LoWo21 eq (11); Graves+ (2008) eq (2); account for both species
    dTdt = 1./m/c_pℓ*(pl.c1.L(zm1m2T[3])*dm1dt + pl.c2.L(zm1m2T[3])*dm2dt- 4.*np.pi*r*pl.K*f1[1]*(zm1m2T[3] - pl.T))
    # return appropriate derivatives
    return np.array([v,dm1dt,dm2dt,dTdt])

# with "standard" assumptions to simplify calculations
def calc_drdt(r,pl):
    '''
    calculate the evaporation (condensation) of the falling rain drop
    change in radius with respect to time
    following Lohmann et al. (2016) eq (7.27) pg 192
    neglecting Kelvin's law and Raoult's law
    makes a series of assumptions handle to handle T_drop
    '''
    # thermodynamic contribution from removing latent heat
    F_thermo = (pl.c.L(pl.T)/pl.c.R/pl.T - 1.)*pl.c.L(pl.T)/pl.K/pl.T*pl.c.ρ(pl.T)
    # diffusion contribution from needing to diffuse away (toward) drop
    F_diff = pl.c.R*pl.T/pl.D_c/pl.c.p_sat(pl.T)*pl.c.ρ(pl.T)
    # combine
    v = drop_prop.calc_v(r,pl,pl.f_rat,pl.f_C_D)
    f = pl.f_vent(r,v,pl)
    drdt = 1./r*(pl.RH-1)/(F_thermo/f[1] + F_diff/f[0])
    return drdt


def calc_drdz(z,r,pl,w=0):
    '''
    calculate the evaporation (condensation) of the falling rain drop
    neglecting Kelvin's law and Raoult's law
    following Lohmann et al. (2016) eq (7.27) pg 192
    neglecting Kelvin's law and Raoult's law
    makes a series of assumptions handle to handle T_drop
    '''
    if r<=1e-7:
        r = np.array([1e-7])
    pl.z2x4drdz(z)
    # thermodynamic contribution from removing latent heat
    F_heat = (pl.c.L(pl.T)/pl.c.R/pl.T - 1.)*pl.c.L(pl.T)/pl.K/pl.T*pl.c.ρ(pl.T)
    # diffusion contribution from needing to diffuse away (toward) drop
    F_mol = pl.c.R*pl.T/pl.D_c/pl.c.p_sat(pl.T)*pl.c.ρ(pl.T)

    v = w-1*drop_prop.calc_v(r[0],pl,pl.f_rat,pl.f_C_D)
    if v>-1e-9:
        v = -1e-9
    f = pl.f_vent(r,v,pl)
    drdt = 1./r*(pl.RH-1)/(F_heat/f[1] + F_mol/f[0])
    return drdt*1./v

##################################################
# FUNCTIONS TO INTEGRATE FALLING
##################################################

def integrate_fall(pl,r0,dr=1e-6,w=0,atol=1e-9,rtol=1e-5,is_lite=False,z_end=None,is_integrate_Tdrop=True):
    '''
    integrate the fall of raindrop from cloud base (z_LCL)
    integrate drdz with option to couple integration with dTdropdz
    use implicit Runge Kutta of order 5 (Radau)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp

    integrate downward in z until drop hits the ground (z = z_end)
    or gets so small that it ceases being a "drop" (r < dr)
    or gets so small that is starts moving upward rather than down (w + v_term > 0)

    all other integrate** functions which follow are essentially the same as this but subtly different
    they should really just be made into one function which better options, but c'est la vie

    inputs:
        * pl [Planet object] - describes planet/atmosphere raindrop lives on
        * r0 [m] - initial raindrop radius
        * (optional) dr [m] - threshold raindrop radius for 'total evaporation'
        * (optional) w [m/s] - vertical velocity
        * (optional) atol [] - absolute tolerance, see scipy ODE solver
        * (optional) rtol [] - relative tolerance, see scipy ODE solver
        * (optional) is_lite [boolean] - whether or not to generate dense_output while integrating
        * (optional) z_end [m] - altitude to end falling, otherwise set from how Planet variables are defined
        * (optional) is_integrate_Tdrop [boolean] - whether to integrate T_drop(z)

    output:
        * sol [output of ivp solver] - see above documentation for properties

    '''
    # set up conditions to end integration
    # drop gets smaller than evaporation threshold dr
    # (dr doesn't equal 0 for numerical stability)
    def evap0(t,y):
        return y[0] - dr
    evap0.terminal = True

    # drop begins moving upward instead of falling
    def v0(t,y):
        return w-1*drop_prop.calc_v(y[0],pl,pl.f_rat,pl.f_C_D)+1e-7
    v0.terminal = True

    # if no vertical wind, don't include second condition
    if w==0:
        events = [evap0]
    else:
        events = [evap0,v0]

    # integrate.Radau requires as input:
    # fun, t0, y0, t_bound
    # where y is being integrated over t given fun of dy/dt starting from y = y0 and t = t0
    # must use functools and lambda function set up because of how scipy's OdeSolver class
    # is written and how numerical Jacobian is calculated for implicit solvers
    if is_integrate_Tdrop:
        radau_drdz = functools.partial(calc_drdz_dTdropdz,pl=pl,w=w)
        y_initial_conditions = np.array([r0,pl.T_LCL])
    else:
        radau_drdz = functools.partial(calc_drdz,pl=pl,w=w)
        y_initial_conditions = np.array([r0])

    # if z to end integration at not specified, make assumptions based on
    # whether z inputted at LCL (gas planet) or surface (terrestrial)
    # also specify first step (letting scipy auto-calculate sometimes leads to errors
    # because of how atmospheric functions are defined)
    if z_end==None:
        if pl.z_LCL==0.:
            first_step = pl.calc_H(0.)/5.
            z_end = -pl.R
        else:
            first_step = pl.z_LCL/5.
            z_end = 0.
    else:
        if pl.z_LCL==0.:
            if z_end > 0:
                z_end = -z_end
        first_step = abs(pl.z_LCL-z_end)/5.

    try:
        sol = integrate.solve_ivp(lambda t,y: radau_drdz(t,y), (pl.z_LCL,z_end), y_initial_conditions, method='Radau', dense_output=not(is_lite), events=events,atol=atol,rtol=rtol,first_step=first_step)
    except ValueError: # if integration fails, start again with stricter tolerance
        print('value error... continuing with lower tol')

        sol = integrate.solve_ivp(lambda t,y: radau_drdz(t,y), (pl.z_LCL,z_end), y_initial_conditions, method='Radau', dense_output=not(is_lite), events=events,atol=atol*1e-2,rtol=rtol*1e-2,first_step=first_step*0.01)
    return sol


def integrate_fall_w_t(pl,r0,dr=1e-6,w=0,atol=1e-9,rtol=1e-5,is_lite=False,is_integrate_Tdrop=True):
    '''
    integrate the fall of raindrop from cloud base (z_LCL)
    integrate dtdz and drdz with option to couple integration with dTdropdz
    y[0] = t, y[1] = r
    '''

    def evap0(t,y):
        return y[1] - dr
    evap0.terminal = True

    def v0(t,y):
        return w-1*drop_prop.calc_v(y[1],pl,pl.f_rat,pl.f_C_D)+1e-7
    v0.terminal = True

    if w==0:
        events = [evap0]
    else:
        events = [evap0,v0]

    if is_integrate_Tdrop:
        radau_dtdz_drdz = functools.partial(calc_dtdz_drdz_dTdz,pl=pl,w=w)
        y_initial_conditions = np.array([0.,r0,pl.T_LCL])
    else:
        radau_dtdz_drdz = functools.partial(calc_dtdz_drdz,pl=pl,w=w)
        y_initial_conditions = np.array([0.,r0])
    if pl.z_LCL==0.:
        first_step = pl.calc_H(0.)/5.
        z_end = -pl.R
    else:
        first_step = pl.z_LCL/5.
        z_end = 0.
    try:
        sol = integrate.solve_ivp(lambda t,y: radau_dtdz_drdz(t,y), (pl.z_LCL,z_end), y_initial_conditions, method='Radau', dense_output=not(is_lite), events=events,atol=atol,rtol=rtol,first_step=first_step)
    except ValueError: # if integration fails, start again with stricter tolerance
        print('value error... continuing with lower tol')
        sol = integrate.solve_ivp(lambda t,y: radau_dtdz_drdz(t,y), (pl.z_LCL,z_end), y_initial_conditions, method='Radau', dense_output=not(is_lite), events=events,atol=atol*1e-2,rtol=rtol*1e-2,first_step=first_step)

    return sol


def integrate_fall_w_t_w_3pl(pl,pl_v,pl_F,r0,dr=1e-6,w=0,atol=1e-9,rtol=1e-5,is_lite=False):
    '''
    integrate the fall of raindrop from cloud base (z_LCL)
    integrate dtdz and drdz with option to couple integration with dTdropdz
    track three different Planet object to account for different aspects of composition in raindrop falling and evaporation
    y[0] = t, y[1] = r
    (new) inputs:
        * pl [Planet object] - governing atmospheric structure (H in LoWo21 Table 2)
        * pl_F [Planet object] - governing molecule and heat transport from drop (transport in LoWo21 Table 2)
        * pl_v [Planet object] - governing raindrop velocity (v_T in LoWo21 Table 2)
    '''


    def evap0(t,y):
        return y[1] - dr
    evap0.terminal = True

    def v0(t,y):
        return w-1*drop_prop.calc_v(y[1],pl,pl.f_rat,pl.f_C_D)+1e-7
    v0.terminal = True

    if w==0:
        events = [evap0]
    else:
        events = [evap0,v0]

    radau_dtdz_drdz = functools.partial(calc_dtdz_drdz_3pl,pl=pl,pl_F=pl_F,pl_v=pl_v,w=w)

    if pl.z_LCL==0.:
        first_step = pl.calc_H(0.)/5.
        z_end = -pl.R
    else:
        first_step = pl.z_LCL/5.
        z_end = 0.
    try:
        sol = integrate.solve_ivp(lambda t,y: radau_dtdz_drdz(t,y), (pl.z_LCL,z_end), np.array([0.,r0,pl.T_LCL]), method='Radau', dense_output=not(is_lite), events=events,atol=atol,rtol=rtol,first_step=first_step)
    except ValueError: # if integration fails, start again with stricter tolerance
        print('value error... continuing with lower tol')
        sol = integrate.solve_ivp(lambda t,y: radau_dtdz_drdz(t,y), (pl.z_LCL,z_end), np.array([0.,r0,pl.T_LCL]), method='Radau', dense_output=not(is_lite), events=events,atol=atol*1e-2,rtol=rtol*1e-2,first_step=first_step)

    return sol

def integrate_fall_w_a(pl,r0,dr=1e-6,atol=1e-9,rtol=1e-5,is_lite=False,f_v0_v_term=1.):
    '''
    integrate the fall of raindrop from cloud base (z_LCL) wrt t, with v(t) not v_T
    integrate drdt, dzdt, dvdt, dTdt
    y[0] = r [m], y[1] = z [m], y[2] = v [m/s], y[3] = T_drop [K]
    (new) inputs:
        * f_v0_v_term [] - fraction of r0 terminal velocity set initial v condition
    '''

    def evap0(t,y):
        return y[0] - dr
    evap0.terminal = True

    def z0(t,y):
        return y[1]
    z0.terminal = True

    radau_drzvdt_w_a = functools.partial(calc_drzvdt_w_a,pl=pl)

    # because integrating in t, not z, need to make an "event" for when z=0 if planet has a surface to stop integration
    # see scipy doc for how events + t_span work in detail
    if pl.z_LCL==0.:
        events = [evap0]
    else:
        events = [evap0,z0]
    pl.z2x4drdz(pl.z_LCL)
    # inital terminal velocity
    # drop will start with specified fraction of this
    v_term_r0 = -drop_prop.calc_v(r0,pl,pl.f_rat,pl.f_C_D) # [m/s]
    first_step = 1 # [s]
    # set integration to end if nothing happens after 2e4 seconds
    t_end = 2e4 # [s]
    try:
        sol = integrate.solve_ivp(lambda t,y: radau_drzvdt_w_a(t,y), (0,t_end), np.array([r0,pl.z_LCL,f_v0_v_term*v_term_r0,pl.T_LCL]), method='Radau', dense_output=not(is_lite), events=events,atol=atol,rtol=rtol,first_step=first_step,max_step=30.)
    except ValueError: # if integration fails, start again with stricter tolerance
        print('value error... continuing with lower tol')
        sol = integrate.solve_ivp(lambda t,y: radau_drzvdt_w_a(t,y), (pl.z_LCL,z_end), np.array([r0,pl.z_LCL,f_v0_v_term*v_term_r0,pl.T_LCL]), method='Radau', dense_output=not(is_lite), events=events,atol=atol*1e-2,rtol=rtol*1e-2,first_step=first_step,max_step=10.)

    return sol


def integrate_2_vterm_w_a(pl,r0,dr=1e-6,atol=1e-9,rtol=1e-5,is_lite=False,f_v0_v_term=0.,f_vterm=1.):
    '''
    integrate falling drop from cloud base (z_LCL) wrt t, with v(t) not v_T
    evaporation is turned off because simply trying to account for v_T assumption effect
    y[0] = z [m], y[1] = v [m/s]
    (new) inputs:
        * f_v0_v_term [] - fraction of r0 terminal velocity set initial v condition
        * f_vterm [] - fraction of terminal velocity for drop to reach to end integration
    '''

    def z0(t,y):
        return y[0]
    z0.terminal = True


    def v_vterm(t,y,pl,f_vterm):
        pl.z2x4drdz(y[0])
        v_term = -1*drop_prop.calc_v(pl.r,pl,pl.f_rat,pl.f_C_D)
        return y[1] - f_vterm*v_term

    event_v_vterm = functools.partial(v_vterm,pl=pl,f_vterm=f_vterm)
    event_v_vterm.terminal = True

    radau_drzvdt_w_a = functools.partial(calc_dzdt_w_a,pl=pl)

    # integrate forward in time until drop reaches inputted fraction of terminal velocity
    # or hits z = 0 if integrating for terrestrial planet
    if pl.z_LCL==0.:
        events = [event_v_vterm]
    else:
        events = [event_v_vterm,z0]
    pl.z2x4drdz(pl.z_LCL)
    v_term_r0 = -drop_prop.calc_v(r0,pl,pl.f_rat,pl.f_C_D)
    pl.r = r0
    first_step = 10.
    t_end = 1e4
    try:
        sol = integrate.solve_ivp(lambda t,y: radau_drzvdt_w_a(t,y), (0,t_end), np.array([pl.z_LCL,f_v0_v_term*v_term_r0]), method='Radau', dense_output=not(is_lite), events=events,atol=atol,rtol=rtol,first_step=first_step)
    except ValueError: # if integration fails, start again with stricter tolerance
        ('value error... continuing with lower tol')
        sol = integrate.solve_ivp(lambda t,y: radau_drzvdt_w_a(t,y), (0,t_end), np.array([pl.z_LCL,f_v0_v_term*v_term_r0]), method='Radau', dense_output=not(is_lite), events=events,atol=atol*1e-2,rtol=rtol*1e-2,first_step=first_step,max_step=10.)

    return sol


def integrate_fall_w_t_T_2condensibles(pl,m10,m20,z_start,dr=1e-6,w=0,atol=1e-9,rtol=1e-5,is_lite=False):
    '''
    integrate the fall of raindrop from cloud height wrt t
    consider mixture of two condensibles
    presently will only work for 1 = N2, 2 = CH4 under Titan conditions
    calculate dzdt, dm1dt, dm2dt, dTdt
    y[0] = z [m], y[1] = m1 [kg], y[2] = m2 [kg], y[3] = T_drop [K]
    (new) inputs:
        * pl [Planet object] - must have c1, c2 condensibles defined
        * m10 [kg] - initial raindrop mass component 1
        * m10 [kg] - initial raindrop mass component 2
        * z_start [m] - altitude at which to begin integration
    output:
        * sol
    '''

    def evap0(t,y):
        m = y[1] + y[2] # [kg] mass of drop
        pl.c.ρ_avg =  (y[1]/m/pl.c1.ρ(y[3]) + y[2]/m/pl.c2.ρ(y[3]))**(-1)
        r = (m/pl.c.ρ_avg*3./4/np.pi)**(1./3)
        return r - dr
    evap0.terminal = True

    def z0(t,y):
        return y[0]
    z0.terminal = True

    # integrate forward in time until drop hits the ground
    # or gets so small that it ceases being a "drop" (r < dr)
    events = [evap0,z0]


    radau_dtdz_dm12dt_dTdropdt = functools.partial(calc_dtdz_dm12dt_dTdropdt,pl=pl,w=w)

    first_step = 20.
    pl.z2x4drdz_12(z_start)

    try:
        sol = integrate.solve_ivp(lambda t,y: radau_dtdz_dm12dt_dTdropdt(t,y), (0,1e5), np.array([z_start,m10,m20,pl.T]), method='Radau', dense_output=not(is_lite), events=events,atol=atol,rtol=rtol,first_step=first_step)
    except ValueError: # if integration fails, start again with stricter tolerance
        pl.z2x4drdz(z_start)
        print('value error... continuing with lower tol')
        sol = integrate.solve_ivp(lambda t,y: radau_dtdz_dm12dt_dTdropdt(t,y), (0,1e5), np.array([z_start,m10,m20,pl.T]), method='Radau', dense_output=not(is_lite), events=events,atol=atol*1e-2,rtol=rtol*1e-2,first_step=first_step)
    return sol

##################################################
# FUNCTIONS FOR SMALLEST RAINDROP (r_min in LoWo21)
##################################################

def calc_smallest_raindrop(pl,dr=1e-6,w=0,z_end=None,is_integrate_Tdrop=True):
    '''
    determine smallest raindrop size which survives to surface
    above some minimum size
    via bisection (woot high tech)
    see LoWo21 Section 3.2
    note the terminology here differs from LoWo21 (r_smallest here = r_min in LoWo21)
    inputs:
        * pl [Planet object]
        * (optional) dr [m] - minimum raindrop radius before considered fully evaporated,
                   also Δr at which to check different raindrop sizes, assumed to be 1 μm
        * (optional) z_end [m] - threshold vertical distance for smallest raindrop to reach,
                                default is None, handled in integrate_fall depending on planetary conditions
    outputs:
        * r_smallest [m] - smallest raindrop size at cloud deck which survives to surface
        * int_descrip [int] - integer to signal potential edge cases
    '''
    r_smallest = None
    int_descrip = None
    r_max = None
    r_min = None
    # maximum stable raindrop size
    # from Rayleigh-Taylor instability, ℓ_RT = 0.5πa
    r_max = drop_prop.calc_r_max_RT(pl,np.pi/2.)
    # smallest r to try
    # if there is vertical wind, adjustments need to be made from using dr
    if v_t_w_0(dr,w,pl)<0:
        r_min = dr
    elif v_t_w_0(r_max,w,pl)>=0:
        return None, -1
    else:
        r_min = brentq(v_t_w_0,dr,r_max,args=(w,pl))

    # confirm atm meets criteria for smallest raindrop to make sense
    is_atm_ok = check_is_atm_ok(pl)
    if not is_atm_ok:
        r_smallest = None
        int_descrip = -3
    else:
        # does smallest threshold value (dr) survive?
        is_r_min_survive = check_is_survive(integrate_fall(pl,r_min,is_lite=True,w=w,z_end=z_end,dr=dr,is_integrate_Tdrop=is_integrate_Tdrop))
        if is_r_min_survive:
            r_smallest = dr
            int_descrip = 1
        else:
            # does maximum threshold value (r_max) survive?
            is_r_max_survive = check_is_survive(integrate_fall(pl,r_max,is_lite=True,w=w,z_end=z_end,dr=dr,is_integrate_Tdrop=is_integrate_Tdrop))
            if not is_r_max_survive:
                r_smallest = None
                int_descrip = -1
            else:
                int_descrip = 0
    # binary search to locate r_smallest
    # (this is slow... but patience is a virtue)
    if int_descrip==0:
        r_smallest_succeed = r_max
        r_largest_fail = dr
        while (r_smallest_succeed - r_largest_fail) > dr:
            r_try = (r_smallest_succeed - r_largest_fail)/2. + r_largest_fail
            is_r_try_survive = check_is_survive(integrate_fall(pl,r_try,is_lite=True,w=w,z_end=z_end,dr=dr,is_integrate_Tdrop=is_integrate_Tdrop))
            if is_r_try_survive:
                r_smallest_succeed = r_try
            else:
                r_largest_fail = r_try
        r_smallest = r_smallest_succeed
    return r_smallest, int_descrip


##################################################
# ANALYZE ODE SOLUTION(S)
##################################################

def check_is_survive(sol):
    '''
    check whether raindrop survived until specified end t or z
    (depending on what integrating)
    input:
        * sol [output of ivp solver]
    output:
        * is_survive [boolean] - whether raindrop reached surface before totally evaporating
    '''
    if sol.status==0:
        is_survive = True
    else:
        is_survive = False
    return is_survive

def calc_last_z(sol):
    '''
    check last altitude integrated to
    input:
        * sol [output of ivp solver]
    output:
        * last_z [s] - last altitude integrated to
    '''
    if sol.status==0:
        last_z = 0.0
    elif sol.status==1:
        try:
            last_z = sol.t_events[0][0]
        except:
            last_z = sol.t_events[1][0]
    else:
        print('error: integration failed',sol.status)
        last_z = None
    return last_z


def calc_last_t(sol):
    '''
    check last time integrated to
    input:
        * sol [output of ivp solver]
    output:
        * last_t [s] - last time integrated to
    '''
    if sol.status==1:
        try:
            last_t = sol.t_events[0][0]
        except:
            last_t = sol.t_events[1][0]
    else:
        print('error: integration failed',sol.status)
        last_t = None
    return last_t

def calc_t2vterm(sol,pl):
    '''
    check time to reach terminal velocity
    input:
        * sol [output of ivp solver]
    output:
        * last_t [s] - time to reach inputted fraction of terminal velocity
    '''

    if sol.status==1:
        try:
            last_t = sol.t_events[0][0]
        except:
            last_t = sol.t_events[1][0]
    elif sol.status==0:
        last_t = 1e4
    else:
        last_t = None
    return last_t

##################################################
# MISC. FUNCTIONS
##################################################

def check_is_atm_ok(pl):
    '''
    check whether cloud deck height is compatible with liquid raindrops
    input:
        * pl [Planet object]
    output:
        * is_atm_ok [boolean] - T/F cloud deck height is consistent with liquid raindrops
    '''
    is_atm_ok = pl.T_LCL >= pl.c.T_freeze
    return is_atm_ok


def v_t_w_0(r,w,pl):
    '''
    zero function for numerical solver
    to calculate r such that terminal velocity of drop of radius r balances against vertical wind
    inputs:
        * r [m] - equiv r
        * w [m/s] - vertical wind speed
        * pl [Planet object]
    output:
        * difference between terminal velocity from inputted r and terminal velocity desired from input w [m/s]
    '''
    return w-1*drop_prop.calc_v(r,pl,pl.f_rat,pl.f_C_D)+1e-7 # (1e-7 gives a cushion to prevent numerical errors)
