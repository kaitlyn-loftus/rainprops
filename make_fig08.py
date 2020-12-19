import matplotlib.pyplot as plt
import numpy as np

# load data
dir = 'output/fig08/'
r_min = np.load(dir+'r_mins_log.npy')
zs = np.load(dir+'zs_log.npy')
zHs = np.load(dir+'zHs_log.npy')
r_maxs = np.load(dir+'r_maxs.npy')
z_maxs = np.load(dir+'z_maxs.npy')
zH_maxs = np.load(dir+'zH_maxs.npy')

# make figure 8
planet_names = ['Earth','early Mars','Jupiter','Saturn','K2-18b']
colors = ['dodgerblue','r','darkorange','gold','0.25']

f, ax = plt.subplots(2,1,figsize=(5,8),sharex=True)
plt.subplots_adjust(hspace=0.05)
for a in ax:
    a.tick_params(right=True,which='both')
    a.tick_params(top=True,which='both')

lw = 1.
for j,pl in enumerate(planet_names):
    ax[0].plot(r_min[:,j,0]*1e3,zs[:,j],lw=2,label=pl,c=colors[j])
    ax[0].axvline(r_maxs[j]*1e3,lw=lw,ls=':',c=colors[j])
ax[0].axvline(100,lw=lw,ls=':',c='0.5',label=r'$r_\mathrm{max}$')
ax[0].set_ylabel(r'$z_\mathrm{LCL} - z$ [m]')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlim(5e-3,10)
ax[0].set_ylim(5e4,10)

for j,pl in enumerate(planet_names):
    ax[1].plot(r_min[:,j,1]*1e3,zHs[:,j],lw=2,label=pl,c=colors[j])
    ax[1].axvline(r_maxs[j]*1e3,lw=lw,ls=':',c=colors[j])
ax[1].set_ylabel(r'$z_\mathrm{LCL} - z$ [$H$]')
ax[1].set_xlabel(r'$r_\mathrm{min}(z)$ [mm]')
ax[1].set_xlim(5e-3,10)
ax[1].set_ylim(1,1e-3)
ax[1].axvline(100,lw=lw,ls=':',c='0.5',label=r'$r_\mathrm{max}$')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[0].legend()
plt.savefig('figs/fig08.pdf',transparent=True,bbox_inches='tight',pad_inches=0.5)
plt.close()
