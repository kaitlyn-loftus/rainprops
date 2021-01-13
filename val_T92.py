################################################################
# check eqs from Thompson+ (1992) are implemented correctly
# in val_G08
# DOI: 10.1016/0019-1035(92)90127-S
################################################################
import numpy as np
import matplotlib as plt
from scipy.optimize import brentq
from val_G08 import calc_γ1,calc_γ2,calc_eq_XN2CH4

# compare against Thompson+ (1992) Table IV
Ts = np.array([94.,93.3,92.6,91.9]) # [K]
ps = np.array([1.5,1.460,1.420,1.390])*1e5 # [Pa]
YCH4 = np.array([0.109,0.103,0.096,0.090]) # [mol/mol], air
XN2 = np.array([0.161,0.167,0.173,0.181]) # [mol/mol], liquid
XCH4 = 1 - XN2
YN2 = 1 - YCH4
pN2s = YN2*ps # [Pa]
pCH4s = YCH4*ps # [Pa ]
γN2 = np.array([1.839,1.830,1.821,1.807]) # [ ]
γCH4 = np.array([1.022,1.024,1.026,1.028]) # [ ]

for i,T in enumerate(Ts):
    print('relative error in γN2',(γN2[i]-calc_γ1(T,XN2[i],XCH4[i]))/γN2[i])
    print('relative error in γCH4',(γCH4[i]-calc_γ2(T,XN2[i],XCH4[i]))/γCH4[i])
    print('relative error in XCH4',(calc_eq_XN2CH4(T,pN2s[i],pCH4s[i])[1]-XCH4[i])/XCH4[i])
    print('relative error in XN2',(calc_eq_XN2CH4(T,pN2s[i],pCH4s[i])[0]-XN2[i])/XN2[i])
    # error in X_i seems due to assuming fugacity coefficient (φ_i) = 1
    # unclear (to KL) how to calculate fugacity coefficient from Thompson+ (1992)...
    # Graves+ (2008) seems to assume φ_i = 1 (φ_i unmentioned)
    print('----')
