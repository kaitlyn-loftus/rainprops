################################################################
# make LoWo21 Figure 3
# r_min, fraction raindrop mass evaporated as functions
# of vertical wind speed
################################################################
import numpy as np
import matplotlib.pyplot as plt

# load results
dir = 'output/fig03/'
ws = np.load(dir+'ws.npy')
r0grid = np.load(dir+'r0grid.npy')
wgrid  = np.load(dir+'wgrid.npy')
m_frac_evap = np.load(dir+'m_frac_evap.npy')
r_mins = np.load(dir+'r_mins.npy')

i_last = -5 # index of last defined r_min

# make figure
plt.figure()
plt.xscale('log')
plt.xlabel(r'$r_0$ [mm]')
plt.ylabel(r'$w$ [m s$^{-1}$]')
plt.tick_params(right=True,which='both')
plt.tick_params(top=True,which='both')

levels_smooth = np.linspace(0,1,250)
cmesh = plt.contourf(r0grid*1e3, wgrid,m_frac_evap,cmap=plt.cm.binary,vmin=0,vmax=1,levels=levels_smooth)
for c in cmesh.collections:
    c.set_edgecolor('face')
    c.set_linewidth(1e-8)
cb = plt.colorbar()
cb.solids.set_edgecolor('face')
# for clarity, plot contour of 10% mass evaporated
# splice up to i_last to not have 10% mass evap curve on top of 100% mass evap curve
c_10 = plt.contour(r0grid[:i_last]*1e3, wgrid[:i_last],m_frac_evap[:i_last],colors='indigo',linewidths=1,linestyles='--',levels=[0.1])
cb.add_lines(c_10)
plt.clabel(c_10, c_10.levels, fmt={0.1:'10% mass evaporated'},fontsize=8)
cb.set_label('fraction mass evaporated [ ]')
cb.set_ticks([0,0.1,0.25,0.5,0.75,1])
plt.axhline(0,lw=0.5,ls='--',c='plum')
# adjust undefined r_min values to hatch total evaporation region correctly
r_mins_no_nan = np.copy(r_mins)
r_mins_no_nan[i_last:] = [10,10,10,10,10] # hack to make fill_between work, essentially makes a horizontal line
plt.xlim(np.amin(r0grid)*1e3,np.amax(r0grid)*1e3)
plt.ylim(-10,10)
plt.fill_betweenx(ws,1e-2,(r_mins_no_nan-1e-6)*1e3,edgecolor='k',facecolor='w',hatch='//')
plt.annotate(s='TOTAL \n EVAPORATION',xy=(0.04,2.5),c='k',backgroundcolor='w')
plt.annotate(s=r'$r_\mathrm{min}$',xy=(0.045,-9),c='darkviolet',backgroundcolor='w')
plt.plot(r_mins*1e3,ws,lw=3,c='darkviolet',zorder=10)
plt.savefig('figs/fig03.pdf',transparent=True,bbox_inches='tight',dpi=500)
plt.close()
