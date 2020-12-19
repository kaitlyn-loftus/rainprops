import numpy as np
import matplotlib as plt
from scipy.optimize import brentq

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

def calc_eq_XN2CH4(T,pN2,pCH4):
    XN2 = brentq(X1_0,1e-2,1,args=(T,pN2,pCH4))
    XCH4 = 1 - XN2
    return XN2, XCH4

Ts = np.array([94.,93.3,92.6,91.9])
ps = np.array([1.5,1.460,1.420,1.390])*1e5
YCH4 = np.array([0.109,0.103,0.096,0.090])
XN2 = np.array([0.161,0.167,0.173,0.181])
XCH4 = 1 - XN2
YN2 = 1 - YCH4
pN2s = YN2*ps
pCH4s = YCH4*ps
γN2 = np.array([1.839,1.830,1.821,1.807])
γCH4 = np.array([1.022,1.024,1.026,1.028])

for i,T in enumerate(Ts):
    print(γN2[i]/calc_γ1(T,XN2[i],XCH4[i]))
    print(γCH4[i]/calc_γ2(T,XN2[i],XCH4[i]))
    # print(type(T))
    # print('result',X1_0(XN2[i],T,pN2s[i],pCH4s[i]))
    # print(calc_eq_XN2CH4(T,pN2s[i],pCH4s[i]))
    print('relative error in XCH4',(calc_eq_XN2CH4(T,pN2s[i],pCH4s[i])[1]-XCH4[i])/XCH4[i])
    print('relative error in N2',(calc_eq_XN2CH4(T,pN2s[i],pCH4s[i])[0]-XN2[i])/XN2[i])
    print('----')
    # print(calc_eq_XN2CH4(T,pN2s[i],pCH4s[i])[0]/XN2[i])
