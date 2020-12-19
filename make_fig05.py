import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.planet as planet

# load results
dir = 'output/fig05/'
ps = np.load(dir+'ps.npy')
ρs = np.load(dir+'rhos.npy')
r_maxs = np.load(dir+'r_maxs.npy')

is_see_all_ℓ = False # set to True to see all length scales instead of ranges

# make figure 5
plt.figure()
plt.xlabel(r'$p_\mathrm{air}$ [Pa]')
plt.ylabel(r'$r_\mathrm{max}$ [mm]')
plt.xscale('log')
plt.tick_params(right=True,which='both')
plt.tick_params(top=True,which='both')
lw = 2
# r_max is in meters so multiply by 1e3 to get mm
plt.plot(ps,r_maxs[:,0]*1e3,c='darkviolet',lw=lw,ls='--',label='Craddock & Lorenz (2017)')
plt.plot(ps,r_maxs[:,1]*1e3,c='forestgreen',lw=lw,ls='-',label='Lorenz (1993)',zorder=10)
plt.plot(ps,r_maxs[:,2]*1e3,c='plum',lw=lw,ls='-',label='Palumbo et al. (2020)',zorder=10)

if is_see_all_ℓ:
    ℓs = [r'FB, $\ell_\sigma = 2\pi r_\mathrm{eq}$',r'FB, $\ell_\sigma = 2\pi a$',r'RT, $\ell_\mathrm{RT} = 0.5\pi a$',r'RT, $\ell_\mathrm{RT} = 2a$',r'RT, $\ell_\mathrm{RT} = 0.5\pi r_\mathrm{eq}$',r'RT, $\ell_\mathrm{RT} = 2 r_\mathrm{eq}$']
    c = ['mediumspringgreen','royalblue','teal','0.75','tomato','yellow']
    for i in range(6):
        plt.plot(ps,r_maxs[:,i+3]*1e3,c=c[i],lw=1,ls=':',label=ℓs[i],zorder=10)
else:
    plt.fill_between(ps,r_maxs[:,6]*1e3,r_maxs[:,7]*1e3,color='0.75',alpha=1,label=r'Rayleigh Taylor')
    plt.fill_between(ps,r_maxs[:,3]*1e3,r_maxs[:,4]*1e3,color='0.25',alpha=1,label='force balance')

plt.vlines(1.1325e5,4,5,colors='dodgerblue',lw=5,label='Earth observations',zorder=102)
plt.vlines(1.1325e5,2.5,5,colors='dodgerblue',lw=5,zorder=99,alpha=0.33)
plt.ylim(0,20)
plt.xlim(1e3,1e7)
plt.legend(ncol=1)
plt.twiny()
plt.xlim(ρs[0],ρs[-1])
plt.xlabel(r'$\rho_\mathrm{air}$ [kg m$^{-3}$]')
plt.xscale('log')
plt.savefig('figs/fig05.pdf',transparent=True,bbox_inches='tight')
plt.close()
