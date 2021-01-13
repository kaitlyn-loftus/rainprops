################################################################
# make LoWo21 Figure 6
# raindrop size bounds across different planetary conditions
################################################################
import numpy as np
import matplotlib.pyplot as plt

# load results
dir = 'output/fig06/'
var_char = np.load(dir+'var_char.npy')
r_mins = np.load(dir+'r_mins.npy')
r_maxs = np.load(dir+'r_maxs.npy')

# adjust fontsizes for figure size
plt.rc('font', size=20)          # change all font sizes
plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize

f, axs = plt.subplots(1,3,sharey=True,figsize=(24,8.5))


alpha = 0.33

q = 5 # index of r_max to use

f, axs = plt.subplots(1,3,sharey=True,figsize=(20,6))
plt.subplots_adjust(wspace=0.1)
axs[0].set_xlabel(r'$p_\mathrm{surf,dry}$ [Pa]')
axs[1].set_xlabel(r'$g$ [m/s$^2$]')
axs[2].set_xlabel(r'$T_\mathrm{surf}$ [K]')

plt.yscale('log')
ylim_down = np.amin(r_mins[:,:,2]*1e3)/1.05
ylim_up = np.amax(r_maxs[:,:]*1e3)*1.05
axs[0].set_ylim(ylim_down,ylim_up)
axs[0].set_xscale('log')
axs[0].set_ylabel(r'$r_\mathrm{eq}$ [mm]')
lw = 3
for i in range(3):
    axs[i].tick_params(right=True,which='both')
    axs[i].tick_params(top=True,which='both')

    axs[i].plot(var_char[i,:],r_maxs[i,:]*1e3,c='0.25',lw=lw,ls='-',label=r'$r_\mathrm{max}$',zorder=101)
    axs[i].plot(var_char[i,:],r_mins[i,:,0]*1e3,color='indigo',lw=lw,label=r'$r_\mathrm{min}$, RH=0.25')
    axs[i].plot(var_char[i,:],r_mins[i,:,1]*1e3,c='darkviolet',lw=lw,ls='-',label=r'$r_\mathrm{min}$, RH=0.5',zorder=10)
    axs[i].plot(var_char[i,:],r_mins[i,:,2]*1e3,color='plum',lw=lw,label=r'$r_\mathrm{min}$, RH=0.75')
    if i==0 or i==1:
        where0 = True
        where1 = True
        where2 = True
        max_bound0 = r_maxs[i,:]*1e3

    elif i==2:
        where0 = r_maxs[i,:]>r_mins[i,:,0]
        where1 = r_maxs[i,:]>r_mins[i,:,1]
        where2 = r_maxs[i,:]>r_mins[i,:,2]
    max_bound1 = np.nanmin([r_mins[i,:,0],r_maxs[i,:]],axis=0)*1e3
    max_bound2 = np.nanmin([r_mins[i,:,1],r_maxs[i,:]],axis=0)*1e3
    max_bound0 = np.nanmin([r_maxs[i,:],r_maxs[i,:]],axis=0)*1e3
    axs[i].fill_between(var_char[i,:],r_mins[i,:,0]*1e3,r_maxs[i,:]*1e3,color='indigo',where=where0,alpha=alpha)
    axs[i].fill_between(var_char[i,:],r_mins[i,:,1]*1e3,max_bound1,color='darkviolet',alpha=alpha)
    axs[i].fill_between(var_char[i,:],r_mins[i,:,2]*1e3,max_bound2,color='plum',alpha=alpha)
    axs[i].set_xlim(np.amin(var_char[i,:]),np.amax(var_char[i,:]))

ylim_down = 1e-5
ylim_up = np.amax(r_maxs[:,:]*1e3)*1.05
axs[0].set_ylim(1e-5,10)
h, l = axs[0].get_legend_handles_labels()
plt.subplots_adjust(bottom=0.275)
f.legend(h, l,loc='lower center',ncol=4,bbox_to_anchor=(0.46,0.05))
# schematically show CCN much smaller than raindrop size bounds
for i in range(3):
    nshad = 20
    CCN_shading = np.zeros((nshad,3))
    CCN_shading[:,0] = np.linspace(0.5,1,nshad)
    CCN_shading[:,1] = np.linspace(0.5,1,nshad)
    CCN_shading[:,2] = np.linspace(0.5,1,nshad)
    newax = f.add_axes(axs[i].get_position(),frameon=True,yticks=[],xticks=[],xlim=[0,2],ylim=[0,1.5])
    newax.imshow(CCN_shading,interpolation='bilinear',cmap=plt.cm.binary,alpha=0.33,extent=(0,2,0,0.55)) #extent=(np.amin(var_char[i,:]),np.amax(var_char[i,:]), 1e-5, 1e-3),
    if i==1:
        newax.annotate(s='CLOUD CONDENSATION NUCLEI SIZES',xy=(0.16,0.27),c='0.25',size=15) #backgroundcolor='w',
plt.savefig('figs/fig06.pdf',transparent=True,bbox_inches='tight',pad_inches=0.75)
