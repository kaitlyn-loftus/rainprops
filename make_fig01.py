import src.planet as planet
import src.fall as fall
import numpy as np
import matplotlib.pyplot as plt
import cycler
import matplotlib as mpl

# load results
dir = 'output/fig01/'
zs = np.load(dir+'zs.npy')
rs = np.load(dir+'rs.npy')
ts = np.load(dir+'ts.npy')

n_r0 = 6
color = ['0.25','0.5','0.75','plum','darkviolet','indigo']
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

plt.figure()
plt.tick_params(right=True,which='both')
plt.tick_params(top=True,which='both')
for i,r0 in enumerate(rs[:,0]):
    plt.plot(rs[i,:]*1e3,zs[i,:],lw=2)
    # add arrow so clearer not a contour plot 
    if zs[i,-1]!=0.0:
        r_arr = np.array([1.02e-5])
        z_arr = zs[i,-3]
        marker = '<'
    else:
        z_arr = zs[i,-8]
        r_arr = rs[i,-8]
        marker = 'v'
    plt.scatter(r_arr*1e3,z_arr,marker=marker)
    plt.annotate(r'$r_0$', xy=(r0*1e3-r0*1e3*0.1, zs[i,0]-17.5),size=8,color=color[i])
    plt.annotate(r'$r_\mathrm{end}$', xy=(r_arr*1e3+r_arr*1e3*0.05, z_arr+5),size=8,color=color[i])
plt.xlabel(r'$r_\mathrm{eq}$($z$) [mm]')
plt.ylabel(r'$z$ [m]')
plt.xscale('log')
plt.xlim(1e-2,rs[-1,0]*1.18e3)
plt.ylim(0,zs[0,0])
plt.savefig('figs/fig01.pdf',transparent=True,bbox_inches='tight')
plt.close()
