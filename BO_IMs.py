from BO_output import bo_output
from Methods import GMSM
output = bo_output()
import numpy as np
Methods = GMSM()
import scipy
import time

####################################################################################
def compute_sim_im(simulation_acc, component, cal_metric, dt, abcissa, damping, duration=False):
    
    metric_sim, Dtm, Dsm, arias = [], [], [], []
    for i in range(len(simulation_acc)):
        acc_ew = simulation_acc[i][1]
        acc_ns = simulation_acc[i][0]
        
        if cal_metric == 'sa':
            # ------------------------------------- #
            ### - Response Spectra
            periods, sa_ew = Methods._responseSpectrum({'acc':acc_ew, 'dt':dt, 'damping':damping, 'periods':abcissa})
            periods, sa_ns = Methods._responseSpectrum({'acc':acc_ns, 'dt':dt, 'damping':damping, 'periods':abcissa})
            gm = (np.array(sa_ew)*np.array(sa_ns))**0.5
            
        elif cal_metric == 'fas':
            # ------------------------------------- #
            ### - Fourier Amplitude Spectra
            freqs, fas_ew = Methods.fast_konno_ohmachi({'acc':acc_ew, 'dt':dt})
            freqs, fas_ns = Methods.fast_konno_ohmachi({'acc':acc_ns, 'dt':dt})
            gm_h = (np.array(fas_ew)*np.array(fas_ns))**0.5
            function = scipy.interpolate.interp1d(freqs, gm_h)
            gm = function(abcissa)

        metric_sim.append(gm)
        
        # - Duration related IMs
        if duration is True:
            ## - Duration - ##
            t, arias_ns = Methods.ariasI({'acc':acc_ns, 'dt':dt, 'units':'cm/s2'})
            index_01 = np.where(arias_ns >= arias_ns.max()*0.01)[0][0]
            index_5 = np.where(arias_ns >= arias_ns.max()*0.05)[0][0]
            index_95 = np.where(arias_ns >= arias_ns.max()*0.95)[0][0]
            index_99 = np.where(arias_ns >= arias_ns.max()*0.99)[0][0]
            Dsm_ns = t[index_95] - t[index_5]
            Dtm_ns = t[index_99] - t[index_01]
            
            t, arias_ew = Methods.ariasI({'acc':acc_ew, 'dt':dt, 'units':'cm/s2'})
            index_01 = np.where(arias_ew >= arias_ew.max()*0.001)[0][0]
            index_5 = np.where(arias_ew >= arias_ew.max()*0.05)[0][0]
            index_95 = np.where(arias_ew >= arias_ew.max()*0.95)[0][0]
            index_99 = np.where(arias_ew >= arias_ew.max()*0.9999)[0][0]
            Dsm_ew = t[index_95] - t[index_5]
            Dtm_ew = t[index_99] - t[index_01]
            Dsm.append((Dsm_ns*Dsm_ew)**0.5)
            Dtm.append((Dtm_ns*Dtm_ew)**0.5)
            
            ## - Arias intensity - ##
            arias.append((arias_ns.max()*arias_ew.max())**0.5)


    return metric_sim, Dtm, Dsm, arias
