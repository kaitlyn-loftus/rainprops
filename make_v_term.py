################################################################
# make LoWo21 Figures S1-S2
# test assumption Δv_raindrop-air = v_term(r) [LoWo21 section 2.2]
# across broad planetary conditions
# for both initial conditions (S1) and due to varying v_term
# as evaporation changes r (S2)
################################################################
import numpy as np
import matplotlib.pyplot as plt

# load results
dir = 'output/sfig01/'
r_maxs = np.load(dir+'r_maxs.npy')
t_vterms = np.load(dir+'t_vterms.npy')
z_vterms = np.load(dir+'z_vterms.npy')
t_t = np.load(dir+'t_vterm_t_evap.npy')
z_z = np.load(dir+'z_vterm_z_evap.npy')
v_error = np.load(dir+'v_error.npy')
z_H_vterm = np.load(dir+'z_H_vterm.npy')


color = ['plum','darkviolet','indigo']
marker = ['*','D','o']
composition = [r'H$_2$',r'N$_2$',r'CO$_2$']
var = [r'$p_\mathrm{LCL}$',r'$T_\mathrm{LCL}$',r'$g$']

# make sfig 1
f, axs = plt.subplots(1,2,figsize=(7.5,5))
plt.subplots_adjust(wspace=0.3)
for j in range(3):
    for x in range(3):
        axs[0].scatter(t_t[j,x,:].flatten(),z_z[j,x,:].flatten(),c=color[x],marker=marker[j],alpha=0.5)
        axs[1].scatter(t_vterms[j,x,:].flatten(),z_H_vterm[j,x,:].flatten(),c=color[x],marker=marker[j],alpha=0.5)
for i in range(3):
    axs[0].scatter(2,2,c='0.75',marker=marker[i],label=composition[i],alpha=0.5)
for i in range(3):
    axs[0].scatter(2,2,c=color[i],marker='X',label=var[i],alpha=0.5)
axs[0].legend()
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_xlim(1e-6,1e-1)
axs[0].set_xlabel(r'$t_{99\%}$ [$t_\mathrm{evap}$]')
axs[0].set_ylabel(r'$z_{99\%}$ [$z_\mathrm{evap}$]')

axs[1].set_xlabel(r'$t_{99\%}$ [s]')
axs[1].set_ylabel(r'$z_{99\%}$ [$H$]')
axs[0].set_ylim(1e-5,1e-1)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlim(1e-2,100)
axs[1].set_ylim(1e-5,1e-1)
plt.savefig('sfigs/sfig01.pdf',transparent=True,bbox_inches='tight',pad_inches=0.5)
plt.close()


# make sfig 2

r0s = np.array([0.05,0.1,0.5,1])

f, axs = plt.subplots(1,2,sharey=True)
plt.subplots_adjust(wspace=0.25)

for j in range(3):
    for x in range(3):
        for i in range(3):
            for k,r0 in enumerate(r0s):
                axs[0].scatter(np.full(v_error[j,x,:,i,k].shape,r0),v_error[j,x,:,1,k].flatten(),c=color[x],marker=marker[j],alpha=0.25)
                axs[1].scatter(np.full(v_error[j,x,:,i,k].shape,r0),v_error[j,x,:,0,k].flatten(),c=color[x],marker=marker[j],alpha=0.25)
for i in range(3):
    axs[0].scatter(10,10,c='0.75',marker=marker[i],label=composition[i],alpha=0.5)
for i in range(3):
    axs[0].scatter(10,10,c=color[i],marker='X',label=var[i],alpha=0.5)
axs[0].legend(ncol=2)

axs[1].set_xscale('log')
axs[0].set_xscale('log')
axs[0].set_xlim(1e-2,4)
axs[1].set_xlim(1e-2,4)
axs[0].set_ylabel(r'median relative error in $v$ [ ]')
axs[1].set_ylabel(r'maximum relative error in $v$ [ ]'+'\n')
axs[1].set_xlabel(r'$r_0$ [mm]')
axs[0].set_xlabel(r'$r_0$ [mm]')
axs[0].set_ylim(1e-4,1)
axs[0].set_yscale('log')
axs[1].set_ylim(1e-4,1)
axs[1].set_yscale('log')
plt.savefig('sfigs/sfig02.pdf',transparent=True,bbox_inches='tight',pad_inches=0.5)
plt.close()
