import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# load results
dir = 'output/fig04/'
r_min = np.load(dir+'r_min.npy')
zs = np.load(dir+'zs.npy')
r0s = np.load(dir+'r0s.npy')
m_frac_evap = np.load(dir+'m_frac_evap.npy')
ℓs_broad = np.load(dir+'ells_broad.npy')
r0s_broad = np.load(dir+'r0s_broad.npy')
percent_err_rmin_broad = np.load(dir+'percent_err_rmin_broad.npy')
percent_err_fevap_broad = np.load(dir+'percent_err_fevap_broad.npy')
rat_rmin_broad = np.load(dir+'rat_rmin_broad.npy')
rat_fevap_broad = np.load(dir+'rat_fevap_broad.npy')
diff_fevap_broad = np.load(dir+'diff_fevap_broad.npy')
r_mins_broad = np.load(dir+'r_mins_broad.npy')
diff_r_mins_broad = r_mins_broad[:,:,:,:,1] - r_mins_broad[:,:,:,:,0]

# set up alignment of Fig 4
fig = plt.figure(figsize=(6,9))
gs = GridSpec(3, 2, figure=fig)
ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[2, 0])
ax3 = fig.add_subplot(gs[2, 1])

plt.subplots_adjust(hspace=0.4)
plt.subplots_adjust(wspace=0.6)
for ax in [ax0,ax1,ax2,ax3]:
    ax.tick_params(right=True,which='both')
    ax.tick_params(top=True,which='both')


ax0.plot(r0s*1e3,m_frac_evap[:,0],lw=2,c='0.75',label='integration')
ax0.plot(r0s*1e3,m_frac_evap[:,1],lw=2,c='darkviolet',ls='--',label=r'min{$\Lambda$,1}')
ax0.scatter(r_min[0,0]*1e3,1.,c='0.75',zorder=10)
ax0.scatter(r_min[0,1]*1e3,1.,c='darkviolet',zorder=10)
ax0.set_xlabel(r'$r_0$ [mm]')
ax0.set_xlim(1e-2,1)
ax0.set_xscale('log')
ax0.set_ylabel('fraction mass evaporated []')
ax0.set_ylim(-0.04,1.04)
ax0.legend(loc=3)

ax1.plot(r_min[:,0]*1e3,zs[:-1],lw=2,c='0.75',label='integration')
ax1.plot(r_min[:,1]*1e3,zs[:-1],lw=2,c='darkviolet',ls='--',label=r'$\Lambda=1$')
ax1.set_xlabel(r'$r_\mathrm{min}(z)$ [mm]')
ax1.set_ylabel(r'$z$ [m]')
ax1.legend(loc=3)
ax1.set_xlim(1e-2,1)
ax1.set_xscale('log')
ax1.set_ylim(-6,zs[-1]+4)

ax3.set_ylabel(r'relative error $r_\mathrm{min}(z_\mathrm{LCL}-\ell)$ [ ]')

ax3.set_xlabel(r'$\ell$ [m]')
for i,ℓ in enumerate(ℓs_broad):
    err_plot = percent_err_rmin_broad[:,:,:,i].flatten()
    ℓ_plot = np.full(err_plot.shape,ℓ)
    ax3.scatter(ℓ_plot,err_plot,color='darkviolet',alpha=0.1)

for i,r0 in enumerate(r0s_broad):
    err_plot = diff_fevap_broad[:,:,:,i].flatten()
    r0_plot = np.full(err_plot.shape,r0)
    ax2.scatter(r0_plot*1e3,err_plot,color='darkviolet',alpha=0.1)
ax2.set_ylabel('difference in fraction\nmass evaporated [ ]')
ax2.set_xscale('log')
ax2.set_ylim(-0.2,0.2)
ax3.set_ylim(-0.2,0.2)
ax3.set_xlim(0,1100)
ax3.axhline(0,ls='--',lw=0.5,c='0.5')
ax2.axhline(0,ls='--',lw=0.5,c='0.5')
ax2.set_xlabel(r'$r_0$ [mm]')
plt.savefig('figs/fig04.pdf',transparent=True,bbox_inches='tight',pad_inches=0.5)
plt.close()
