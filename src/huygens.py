################################################################
# module to convert data files from Huygens mission into
# z-dependent functions
################################################################
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import datetime

def getz2xfunctions():
    '''
    function to process Huygens data from discrete data points to
    continuous functions of altitude z
    use linear interpolation from scipy
    outputs:
        * z2p, z2T, z2fCH4 [functions] to convert altitude to pressure, temperature, and CH4 molar concentration
    '''
    ppi_f = 'data/Huygens_PPI.txt'
    # source: https://atmos.nmsu.edu/PDS/data/hphasi_0001/DATA/PPI/HASI_L4_PPI_PRESSURE_VEL.TAB
    # documentation: https://atmos.nmsu.edu/PDS/data/hphasi_0001/DATA/PPI/HASI_L4_PPI_PRESSURE_VEL.LBL
    tem_f = 'data/Huygens_TEM.txt'
    # source: https://atmos.nmsu.edu/PDS/data/hphasi_0001/DATA/TEM/HASI_L4_TEM_TEMPERATURE.TAB
    # documentation: https://atmos.nmsu.edu/PDS/data/hphasi_0001/DATA/TEM/HASI_L4_TEM_TEMPERATURE.LBL
    gcms_f = 'data/Huygens_GCMS.txt'
    # source: https://atmos.nmsu.edu/PDS/data/hpgcms_0001/DATA/DTWG_MOLE_FRACTION/GCMS_MOLE_FRACTION_STG2.TAB
    # documentation: https://atmos.nmsu.edu/PDS/data/hpgcms_0001/DATA/DTWG_MOLE_FRACTION/GCMS_MOLE_FRACTION_STG2.LBL
    # "Obviously the mole fraction for Nitrogen (N2) is [1. - SUM(MF(CH4)+MF(Ar)+MF(XX))]."
    # => molar concentrations
    ppi_data = np.genfromtxt(ppi_f,delimiter=';')
    tem_data = np.genfromtxt(tem_f,delimiter=';')
    gcms_data = np.genfromtxt(gcms_f,skip_header=1)
    gcms_data = pd.read_csv(gcms_f,header=0,parse_dates=[0])
    # convert UTC_ABS_TIME to same units as descent file
    # START_TIME =  2005-01-14T09:11:21.373 # PPI, TEM

    t0 = np.datetime64('2005-01-14T09:11:21.373')
    t_elapsed_ms = (gcms_data['UTC_ABS_TIME'].values-t0)/np.timedelta64(1, 'ms')
    t_surface =  8878990.
    gcms_data_ms = np.array([t_elapsed_ms,gcms_data['CH4'].values])
    gcms_data_ms = gcms_data_ms.transpose()

    # what's stored in read-in arrays
    # ppi_data[:,0] # time [milliseconds]
    # ppi_data[:,2] # total pressure [Pa]
    # ppi_data[:,4] # z [m]
    # tem_data[:,0] # time [milliseconds]
    # tem_data[:,1] # z [m]
    # tem_data[:,2] # T [K]
    # gcms_data_ms[:,0] # time [milliseconds]
    # gcms_data_ms[:,1] # f_CH4 [mol/mol]

    # only include data before hit surface
    ppi_data = ppi_data[ppi_data[:,0]<=t_surface]
    tem_data = tem_data[tem_data[:,0]<=t_surface]
    gcms_data_ms = gcms_data_ms[gcms_data_ms[:,0]<=t_surface]

    # interpolations to get functions in terms of z
    t2z = interp1d(ppi_data[:,0], ppi_data[:,4],bounds_error=False,fill_value=(ppi_data[0,0],0.))
    t2fCH4 = interp1d(gcms_data_ms[:,0], gcms_data_ms[:,1],bounds_error=False,fill_value=(gcms_data_ms[0,1],gcms_data_ms[-1,1]))
    z2fCH4 = interp1d(ppi_data[:,4],t2fCH4(ppi_data[:,0]),bounds_error=False,fill_value=(gcms_data_ms[-1,1],gcms_data_ms[0,1]))
    z2t = interp1d(ppi_data[:,4],ppi_data[:,0],bounds_error=False,fill_value=(ppi_data[-1,0],ppi_data[0,0]))
    z2T = interp1d(tem_data[:,1], tem_data[:,2],bounds_error=False,fill_value=(tem_data[-1,2],tem_data[0,2]))
    z2p = interp1d(ppi_data[:,4],ppi_data[:,2],bounds_error=False,fill_value=(ppi_data[-1,2],ppi_data[0,2]))

    return z2p, z2T, z2fCH4
