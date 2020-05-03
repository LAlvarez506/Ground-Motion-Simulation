from BO_output import bo_output
output = bo_output()
import time
import numpy as np
from Methods import GMSM
Methods = GMSM()
from Otarola_Parallel import EQ
from EXSIM_Parallel import EQ as EXSIM_EQ

def sim_fas_multiple(Mw,input_parameters,n_sim,component,dt,freqs,b,delta_az,pp):
    # - Simulate the events
    a = EQ(Mw,input_parameters,n_sim=n_sim)
    simulation_acc = a.sim_acc
    r_rup = a.R_rup

    ### - IM
    metric_sim = []
    for i in range(n_sim):
        if component is 'EW':
            acc_sim = simulation_acc[i][1]
            freqs,fas = Methods.fast_konno_ohmachi({'acc':acc_sim,'dt':dt})
            FAS = [np.interp(f,freqs,fas) for f in freqs]
            metric_sim.append(FAS)
        elif component is 'NS':
            acc_sim = simulation_acc[i][0]
            freqs, fas = Methods.fast_konno_ohmachi({'acc':acc_sim,'dt':dt})
            FAS = [np.interp(f,freqs,fas) for f in freqs]
            metric_sim.append(FAS)
        elif component is 'UD':
            acc_sim = simulation_acc[i][2]
            freqs, fas = Methods.fast_konno_ohmachi({'acc':acc_sim,'dt':dt})
            FAS = [np.interp(f,freqs,fas) for f in freqs]
            metric_sim.append(FAS)
        elif component is 'rotD' or component is 'rotI':
            acc_ew = simulation_acc[i][1]
            acc_ns = simulation_acc[i][0]
            
            freqs,RotD,RotI = methods.FAS_rotComp(acc_ew,acc_ns,delta_az,pp,dt)
            if component is 'rotD':
                freqs,fas = methods.fast_konno_ohmachi({'FAS':RotD,'freqs':freqs})
            else:
                freqs,fas = methods.fast_konno_ohmachi({'FAS':RotI,'freqs':freqs})

            FAS = [np.interp(f,freqs,fas) for f in freqs]
            metric_sim.append(FAS)
        elif component is 'GM':
            acc_ew = simulation_acc[i][1]
            acc_ns = simulation_acc[i][0]
            freqs_ew,fas_ew = Methods.fast_konno_ohmachi({'acc':acc_ew,'dt':dt})
            freqs_ns,fas_ns = Methods.fast_konno_ohmachi({'acc':acc_ns,'dt':dt})

            gm = []
            for f in freqs:
                ew = np.interp(f,freqs_ew,fas_ew) 
                ns = np.interp(f,freqs_ns,fas_ns)
                gm.append((ew*ns)**0.5)

            metric_sim.append(gm)

    return simulation_acc,metric_sim,r_rup
####################################################################################
def sim_sa_multiple(Mw,input_parameters,n_sim,component,dt,periods,damping):
    # - Simulate the events
    a = EQ(Mw,input_parameters,n_sim=n_sim)
    simulation_acc = a.sim_acc
    r_rup = a.R_rup

    ### - IM
    metric_sim = []
    for i in range(n_sim):
        if component is 'EW':
            acc_sim = simulation_acc[i][1]
            freqs,fas = Methods.fast_konno_ohmachi({'acc':acc_sim,'dt':dt})
            FAS = [np.interp(f,freqs,fas) for f in freqs]
            metric_sim.append(FAS)
        elif component is 'NS':
            acc_sim = simulation_acc[i][0]
            freqs, fas = Methods.fast_konno_ohmachi({'acc':acc_sim,'dt':dt})
            FAS = [np.interp(f,freqs,fas) for f in freqs]
            metric_sim.append(FAS)
        elif component is 'UD':
            acc_sim = simulation_acc[i][2]
            freqs, fas = Methods.fast_konno_ohmachi({'acc':acc_sim,'dt':dt})
            FAS = [np.interp(f,freqs,fas) for f in freqs]
            metric_sim.append(FAS)
        elif component is 'rotD' or component is 'rotI':
            acc_ew = simulation_acc[i][1]
            acc_ns = simulation_acc[i][0]
            
            freqs,RotD,RotI = methods.FAS_rotComp(acc_ew,acc_ns,delta_az,pp,dt)
            if component is 'rotD':
                freqs,fas = methods.fast_konno_ohmachi({'FAS':RotD,'freqs':freqs})
            else:
                freqs,fas = methods.fast_konno_ohmachi({'FAS':RotI,'freqs':freqs})

            FAS = [np.interp(f,freqs,fas) for f in freqs]
            metric_sim.append(FAS)
        elif component is 'GM':
            acc_ew = simulation_acc[i][1]
            acc_ns = simulation_acc[i][0]
            periods, sa_ew = Methods._responseSpectrum({'acc':acc_ew, 'dt':dt, 'damping':damping, 'periods':periods})
            periods, sa_ns = Methods._responseSpectrum({'acc':acc_ns, 'dt':dt, 'damping':damping, 'periods':periods})

            gm = (np.array(sa_ew)*np.array(sa_ns))**0.5


            metric_sim.append(gm)

    return simulation_acc,metric_sim,r_rup
####################################################################################
def EXSIM_sa_multiple(Mw, input_parameters, n_sim,component, dt, periods, damping):
    # - Simulate the events
    a = EXSIM_EQ(Mw,input_parameters,n_sim=n_sim)
    simulation_acc = a.sim_acc
    r_rup = a.R_rup

    ### - IM
    metric_sim = []
    for i in range(n_sim):
        if component is 'GM':
            acc = simulation_acc[i]
            periods, sa = Methods._responseSpectrum({'acc':acc, 'dt':dt, 'damping':damping, 'periods':periods})
            metric_sim.append(np.array(sa))

    return simulation_acc,metric_sim,r_rup

#################################################################################### 
def sim_spectrum_single(Mw,input_parameters,n_sim,component,dt,periods,damping,b,delta_az,pp):
    # - Simulate the events
    a = EQ(Mw,input_parameters,n_sim=n_sim)
    simulation_acc = a.sim_acc
    r_rup = a.R_rup

    ### - IM
    metric_sim = []
    for i in range(n_sim):
        if component is 'EW':
            acc_sim = simulation_acc[i][1]
            periods,sa = Methods.fast_konno_ohmachi({'acc':acc_sim,'dt':dt,'damping':damping,'periods':periods})
            metric_sim.append(sa)
        elif component is 'NS':
            acc_sim = simulation_acc[i][0]
            periods,sa = Methods.fast_konno_ohmachi({'acc':acc_sim,'dt':dt,'damping':damping,'periods':periods})
            metric_sim.append(sa)
        elif component is 'UD':
            acc_sim = simulation_acc[i][2]
            periods,sa = Methods.fast_konno_ohmachi({'acc':acc_sim,'dt':dt,'damping':damping,'periods':periods})
            metric_sim.append(sa)

        elif component is 'GM':
            acc_ew = simulation_acc[i][1]
            acc_ns = simulation_acc[i][0]
            periods,sa_ew = Methods.fast_konno_ohmachi({'acc':acc_ew,'dt':dt,'damping':damping,'periods':periods})
            periods,sa_ns = Methods.fast_konno_ohmachi({'acc':acc_ns,'dt':dt,'damping':damping,'periods':periods})

            sa_ew = np.array(sa_ew)
            sa_ns = np.array(sa_ns)
            gm = (sa_ew*sa_ns)**0.5
            metric_sim.append(gm)

    return simulation_acc,metric_sim,r_rup
       
