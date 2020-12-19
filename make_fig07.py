import matplotlib.pyplot as plt
import numpy as np

dir = 'output/fig07/'
pl_labels = ['h2','h2-n2_10-90','n2','co2']
pl_labels2 = [r'H$_2$',r'N$_2$',r'CO$_2$']
pl_labels3 = [r'CO$_2$',r'N$_2$',r'H$_2$']
i_r0 = range(3)
r0s = np.load(dir+'r0s.npy')

plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize


ls = ['-','--',':']
c = ['indigo','darkviolet','plum']
f, ax = plt.subplots(3,1,sharex=True,figsize=(6,18))
plt.subplots_adjust(hspace=0.05)
lw = 2

# load results
for i,r0 in enumerate(r0s):
    P_evap_N2 = np.load(dir+'power_evap_norm_'+pl_labels[2]+'_'+str(i)+'.npy')
    P_evap_H2 = np.load(dir+'power_evap_norm_'+pl_labels[0]+'_'+str(i)+'.npy')
    P_evap_CO2 = np.load(dir+'power_evap_norm_'+pl_labels[3]+'_'+str(i)+'.npy')
    P_evaps = [P_evap_H2,P_evap_N2,P_evap_CO2]

    p_evap_N2 = np.load(dir+'p_evap_norm_'+pl_labels[2]+'_'+str(i)+'.npy')
    p_evap_H2 = np.load(dir+'p_evap_norm_'+pl_labels[0]+'_'+str(i)+'.npy')
    p_evap_CO2 = np.load(dir+'p_evap_norm_'+pl_labels[3]+'_'+str(i)+'.npy')
    p_evaps = [p_evap_H2,p_evap_N2,p_evap_CO2]

    ts_evap_N2 = np.load(dir+'ts_evap_'+pl_labels[2]+'_'+str(i)+'.npy')
    ts_evap_H2 = np.load(dir+'ts_evap_'+pl_labels[0]+'_'+str(i)+'.npy')
    ts_evap_CO2 = np.load(dir+'ts_evap_'+pl_labels[3]+'_'+str(i)+'.npy')
    ts_evap = [ts_evap_H2,ts_evap_N2,ts_evap_CO2]

    zs_evap_N2 = np.load(dir+'zs_evap_'+pl_labels[2]+'_'+str(i)+'.npy')
    zs_evap_H2 = np.load(dir+'zs_evap_'+pl_labels[0]+'_'+str(i)+'.npy')
    zs_evap_CO2 = np.load(dir+'zs_evap_'+pl_labels[3]+'_'+str(i)+'.npy')
    zs_evap = [zs_evap_H2,zs_evap_N2,zs_evap_CO2]



    for j in range(3):
        ax[0].plot(-P_evaps[j],p_evaps[j][1:],lw=lw,c=c[j],ls=ls[i])
        ax[2].plot(-P_evaps[j],ts_evap[j][1:],lw=lw,c=c[j],ls=ls[i])
        ax[1].plot(-P_evaps[j],-zs_evap[j][1:]/1e3,lw=lw,c=c[j],ls=ls[i])

for j,lab in enumerate(pl_labels2):
    ax[0].plot(2,2,label=lab,c=c[j],lw=lw,ls='-')
for j,r0 in enumerate(r0s):
    ax[0].plot(2,2,label=r'$r_0$=%1.1f mm'%(r0*1e3),c='0.75',lw=lw,ls=ls[j])

ax[0].set_xlim(0,-0.2)
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$p$ [Pa]')
ax[0].set_ylim(2.4e5,7.5e4)
ax[2].set_xlim(0,-0.225)
ax[2].set_ylim(1200,0)
ax[2].set_ylabel(r'$t$ [s]')
ax[2].set_xlabel(r'$P_\mathrm{evap}$ [W]')
ax[1].set_ylim(30,0)
ax[0].legend()

ax[1].set_ylabel(r'$z_\mathrm{LCL} - z$ [km]')

ax[0].tick_params(top=True,which='both')
ax[0].tick_params(right=True,which='both')
ax[1].tick_params(top=True,which='both')
ax[1].tick_params(right=True,which='both')
ax[2].tick_params(top=True,which='both')
ax[2].tick_params(right=True,which='both')

plt.savefig('figs/fig07.pdf',transparent=True,bbox_inches='tight',pad_inches=0.75)
plt.close()
