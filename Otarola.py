# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 06:11:20 2019

@author: User
"""

""" Modificqtion regarding last version:
        - Completly modified to accept only one faults plane
"""
 ####################-----------------------############################      
 ##################------- EARTHQUAKE --------##########################  
 ####################-----------------------############################
import sys
wd = ''
from Source import source
from scipy.optimize import least_squares
from scipy import signal as ScipySignal
from scipy.fftpack import fft
from scipy.fftpack import ifft
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import math as m
import time
import scipy
import random
from Methods import GMSM
Methods = GMSM()
class EQ(object):

    # - Medium
    correlation = False
    site_coord = None
    Hypocenter = None
    Betha = None
    Alpha = None
    vs = None
    vp = None
    Rho=None
    Depth=None
    v_rupture = 0.80

    # - Duration
    random_seed_boolean = False
    random_seed = None
    path_duration = [(lambda R:R*0.05)]
    path_duration_lim = [1000.0]
    source_duration = (lambda f:1.0/f)
    lag_max = 0.0
    Tgm_max = 0.0
    Tgm = 0.0
    boolean_tgm = False
    dt = 0.01
    samples_t = 0
    t = []
    freq_ps = np.logspace(np.log10(0.001),np.log10(100.), 1000)
    fc = 0.0
    fc_baseline = None
    order_PostProcess = 3

    # - Window signal
    window = {}
    window['epsilon_window'] = 0.2
    window['nu_window'] = 0.2
    window['f_tgm'] = 2.0
    window['window_type'] = 'Saragoni'

    # - Source
    ## - Geometry
    L,W = None,None
    nl,nw = 2,2
    fault_type = 'All'
    fault_geom_Mw = None
    rigidity = 32e9
    rake = 0.0
    dip = 90.0
    random_strike = None
    random_dip = None
    hyp_L = 0.5
    hyp_W = 0.5
    
    ## - Parameters
    stress_drop = 150 #bar
    F_pulse = 50.0
    velocity_ratio = 0.57
    Mo = 0.0
    hypocenter = []
    site_coord = []
    N_psources = 0.0
    
    ## - Stochastic field
    slip_dist = 'vonkarman'
    slip_mean = 0.256
    slip_cov = 0.40
    a_x = 5.0
    a_z = 2.0
    H = 0.2

    vertices = None
    hyp_rel_bool = None
    hyp_coord = None
    hyp_geo_rel_L = None
    hyp_geo_rel_W = None
    plots_boolean = False
    L_Mw_factor,W_Mw_factor = None,None
    fault_geom_Mw_uncertainty = True
    fault_origin = None

    # - The path 
    geom_spread_lim = None
    b1 = None
    b2 = None
    Qo = None
    Q1 = None
    Q1exp = None

    # - The site
    amp_type = 'elastic'
    amp_funct = None
    amp_freq = freq_ps
    kappa = 0.03
    kappa_type = None
    shift_freq = None
    f_max = None

    # - Signal processing
    signal_treatment = True
    
    # - Information print
    print_info = False

    # - Multiprocessing
    n_processors = 4
    
    def __init__(self,geometry,Mw,parameters=None,wd=None,print_info=None):
        t0 = time.time()
        
        if print_info is not None:
            EQ.print_info = print_info
            
        if wd is not None:
            EQ.wd = wd        
        
        self.Mw = Mw
        EQ.Mo = 10**((Mw + 10.74)*3.0/2.0)
        
        # - Propagation Medium
        if 'Depth' in parameters:
            """
            Soil layer depth, in m
            """
            if parameters['Depth'] is not None:
                EQ.Depth = parameters['Depth']  
                if 'vs' in parameters:
                    """
                    Velocity profile (S-waves) for the soil layer depth array, in m/2
                    """
                    if parameters['vs'] is not None:
                        EQ.vs = parameters['vs']
                        if len(EQ.vs) != len(EQ.Depth):
                            print('Mismatch in depth and vs dimension list')
                        
                if 'vp' in parameters:
                    """
                    Velocity profile (P-waves) for the soil layer depth array, in m/2
                    """
                    if parameters['vp'] is not None:
                        EQ.vp = parameters['vp'] 
                        if len(EQ.vp) != len(EQ.Depth):
                            print('Mismatch in depth and vp dimension list')
                        
        if 'velocity_ratio' in parameters:
            """
            Ratio S/P wave propagation velocity
            """
            if parameters['velocity_ratio'] is not None:
                EQ.velocity_ratio = parameters['velocity_ratio']
        if 'Rho' in parameters:
            """
            Density at the hypocenter, in g/cm3
            """
            if parameters['Rho'] is not None:
                EQ.Rho = parameters['Rho']    
        if 'Betha' in parameters:
            """
            Shear wave propafation velocity at the hypocenter, in km/s
            """
            if parameters['Betha'] is not None:
                EQ.Betha = parameters['Betha']   
                
        if 'Alpha' in parameters:
            """
            P wave propafation velocity at the hypocenter, in km/s
            """
            if parameters['Alpha'] is not None:
                EQ.Alpha = parameters['Alpha']  
        
        if 'correlation' in parameters:
            if parameters['correlation'] is not None:
                EQ.correlation = parameters['correlation']  
            if 'df' in parameters:
                if parameters['df'] is not None:
                    EQ.df = parameters['df']
                
        if 'dt' in parameters:
            if parameters['dt'] is not None:
                EQ.dt = parameters['dt']
            
        if 'v_rupture' in parameters:
            """
            Rupture velocity as a fraction of Betha
            """
            if parameters['v_rupture'] is not None:
                x = parameters['v_rupture']
                EQ.v_rupture = x*EQ.Betha

        # - Geometry
        EQ.coordinate_system = geometry['system']
        if EQ.coordinate_system is 'local':
            pass

        else:
            """
            Y axis: N-S (Latitude)
            X axis: E-W (Longitude)
            Z axis: vertical
            """
            
           # - Fault discretization 
            if 'nw' in geometry:
                EQ.nw = geometry['nw']
            if 'nl' in geometry:
                EQ.nl = geometry['nl']
                
            """
            Minimun discretization for finite fault is 2
            """
            if EQ.nw == 1 and EQ.nl == 1:
                pass
            else:
                if EQ.nw%2 == 1:
                    EQ.nw += 1
                
                if EQ.nl%2 == 1:
                    EQ.nl += 1
            
            if 'strike' in geometry:
                if geometry['strike'] == 0.0:
                    EQ.strike = 0.0001
                else:
                    EQ.strike = geometry['strike']
            EQ.dip = geometry['dip']
            if 'random_strike' in geometry:
                if geometry['random_strike'] is True:
                    EQ.strike = random.uniform(0.0001,360.0)
            if 'random_dip' in geometry:
                if geometry['random_dip'] is True:
                    EQ.dip = random.uniform(10.0,90.0)
            
            if 'L' in geometry and 'W' in geometry:
                EQ.L = geometry['L']
                EQ.W = geometry['W']
            else:
                if 'fault_type' in geometry:
                    fault_type = geometry['fault_type']
                else:
                    fault_type = None
                EQ.L,EQ.W = Methods.fault_geometry_Mw(Mw,fault_type=fault_type)

            # - Hypocenter
            if 'hyp_L' in geometry:
                EQ.hyp_L = geometry['hyp_L']
            if 'hyp_W' in geometry:
                EQ.hyp_W = geometry['hyp_W']
                
            # - Construct the fault - #
            self.source = source()       
            self.source.create_plane(EQ.L,EQ.W,EQ.nl,EQ.nw)
            fault_corners = self.source.corners
            sub_sources_centroid = self.source.sub_sources_centroid
           
           # - Hypocenter in source local coordinates (not rotated)
            hyp_local = np.array([EQ.hyp_L*EQ.L,0.0,EQ.hyp_W*EQ.W])
  
            # - Rupture time and activation order
            self.source.rupture_time_activation(EQ.v_rupture,hyp_local)
            self.sub_source_trup =  self.source.t_rupture
            self.sub_source_activation =  self.source.activation_index
            
            # - Rotate the fault to align with dip and strike
            ## - Dip
            corners_dip = self.source.rotate_plane(fault_corners,EQ.dip,'x')
            sub_sources_centroid_dip = self.source.rotate_plane(sub_sources_centroid,EQ.dip,'x')
            hyp_dip = self.source.rotate_plane(hyp_local,EQ.dip,'x')

            ## - Strike
            corners_aligned = self.source.rotate_plane(corners_dip,EQ.strike,'z')
            sub_sources_centroid_aligned = self.source.rotate_plane(sub_sources_centroid_dip,EQ.strike,'z')
            hyp_aligned = self.source.rotate_plane(hyp_dip,EQ.strike,'z')
            
            # - Update the Z coordinate based on the real hypocenter depth
            if 'hypocenter' in geometry:
                if geometry['hypocenter'] is not None:
                    dz = geometry['hypocenter'][2] - hyp_aligned[2]
                    EQ.Hypocenter = [hyp_aligned[0],hyp_aligned[1],hyp_aligned[2]+dz]
                    EQ.site_coord = Methods.locate_site(geometry['hypocenter'],geometry['site'],reference=EQ.Hypocenter)                        
                    EQ.vertices = [[p[0],p[1],p[2]+dz] for p in corners_aligned]
                    EQ.sub_fault_centroid = [[p[0],p[1],p[2]+dz] for p in sub_sources_centroid_aligned]
                else:
                    print('Hypocenter not specified')
            else:
                print('Hypocenter not specified')

        # - Parameters - #
        if 'window' in parameters:
            if parameters['window'] is not None:
                if 'type' in parameters['window']:
                    EQ.window['window_type'] = parameters['window']['type']
                if 'epsilon' in parameters['window']:
                    EQ.window['epsilon_window'] = parameters['window']['epsilon']
                if 'nu' in parameters['window']:
                    EQ.window['nu_window'] = parameters['window']['nu']
                if 'f_tgm' in parameters['window']:
                    EQ.window['f_tgm'] = parameters['window']['f_tgm']


        if 'source' in parameters:
            if parameters['source'] is not None:
                if 'slip_dist' in parameters['source']:
                    EQ.slip_dist = parameters['source']['slip_dist']
                if 'a_x' in parameters['source']:
                    EQ.a_x = parameters['source']['a_x']
                if 'a_z' in parameters['source']:
                    EQ.a_z = parameters['source']['a_z']
                if 'H' in parameters['source']:
                    EQ.H = parameters['source']['H']               
                if 'slip_mean' in parameters['source']:
                    EQ.slip_mean = parameters['source']['slip_mean']  
                if 'slip_cov' in parameters['source']:
                    EQ.slip_cov = parameters['source']['slip_cov']                      

        if 'Tgm' in parameters:
            if parameters['Tgm'] is not None:
                EQ.Tgm = parameters['Tgm']
                EQ.boolean_tgm = True
        if 'path_duration' in parameters:
            if parameters['path_duration'] is not None:
                EQ.path_duration = parameters['path_duration']
        if 'path_duration_lim' in parameters:
            if parameters['path_duration_lim'] is not None:
                EQ.path_duration_lim = parameters['path_duration_lim']
        if 'source_duration' in parameters:
            if parameters['source_duration'] is not None:
                EQ.source_duration = parameters['source_duration']                
        if 'F_pulse' in parameters:
            if parameters['F_pulse'] is not None:
                EQ.F_pulse = parameters['F_pulse']
        if 'stress_drop' in parameters:
            if parameters['stress_drop'] is not None:
                EQ.stress_drop = parameters['stress_drop']
        
        # - Geometrical spread
        if 'b1' in parameters:
            if parameters['b1'] is not None:
                EQ.b1 = parameters['b1']
        if 'b2' in parameters:
            if parameters['b2'] is not None:
                EQ.b2 = parameters['b2']
            if parameters['geom_spread_lim'] is not None:
                EQ.geom_spread_lim = parameters['geom_spread_lim']
        if 'Qo' in parameters:
            if parameters['Qo'] is not None:
                EQ.Qo = parameters['Qo']
        if 'Q1' in parameters:
            if parameters['Q1'] is not None:
                EQ.Q1 = parameters['Q1']
        if 'Q1exp' in parameters:
            if parameters['Q1exp'] is not None:
                EQ.Q1exp = parameters['Q1exp']
        
        if 'kappa' in parameters:
            if parameters['kappa'] is not None:
                EQ.kappa = parameters['kappa']
        if 'kappa_type' in parameters:
            if parameters['kappa_type'] is not None:
                EQ.kappa_type = parameters['kappa_type']
            if 'shift_freq' in parameters:
                if parameters['shift_freq'] is not None:
                    EQ.shift_freq = parameters['shift_freq']
                    
        if 'f_max' in parameters:
            if parameters['f_max'] is not None:
                EQ.f_max = parameters['f_max']
        if 'amp_type' in parameters:
            if parameters['amp_type'] is not None:
                EQ.amp_type = parameters['amp_type']
        if 'amp_funct' in parameters:
            if parameters['amp_funct'] is not None:
                EQ.amp_funct = parameters['amp_funct']
        if 'amp_freq' in parameters:
            if parameters['amp_freq'] is not None:
                EQ.amp_freq = parameters['amp_freq']

        if 'signal_treatment' in parameters:
            if parameters['signal_treatment'] is not None:
                EQ.signal_treatment = parameters['signal_treatment']
        if 'fc_baseline' in parameters:
            if parameters['fc_baseline'] is not None:
                EQ.fc_baseline = parameters['fc_baseline'] 
 
        if 'seed' in parameters:
            if parameters['seed'] is not None:
                EQ.random_seed_boolean = True
                EQ.random_seed = parameters['seed']
            else:
                EQ.random_seed = np.random.randint(0,10000.)
        if 'plots_boolean' in parameters:
            if parameters['plots_boolean'] is not None:
                EQ.plots_boolean = parameters['plots_boolean']    

        if 'n_processors' in parameters:
            if parameters['n_processors'] is not None:
                EQ.n_processors = parameters['n_processors']    

        if EQ.Betha is None:
            EQ.Betha = Methods.velocity_profile(EQ.Hypocenter[2])
        if EQ.Alpha is None:
            EQ.Alpha = EQ.Betha/EQ.velocity_ratio
        if EQ.Rho is None:
            EQ.Rho = Methods.density_profile(EQ.Betha)
        
        EQ.fc = 4.9e6*EQ.Betha*(EQ.stress_drop/(EQ.Mo*EQ.F_pulse/100.0))**(1.0/3.0)

        # - Get hypocentral and epicentral distances
        self.R_hypocentral = np.linalg.norm(np.array(EQ.site_coord) - np.array(EQ.Hypocenter))
        self.R_epicentral = np.linalg.norm(np.array(EQ.site_coord) - np.array([EQ.Hypocenter[0],EQ.Hypocenter[1],0.0]))
        self.R_rup,self.R_jb = Methods.compute_rupture_distances(EQ.vertices,EQ.site_coord,n=20)
        
        # - Print info regarding geometrical layout
        if EQ.print_info == True:
            print('########### - GENERAL - #################')
            print('Mw = ',Mw)
            print('Geometry input system = ',geometry['system'])
            print('Generate plots = ',EQ.plots_boolean)
            print('Signal treatment = ',EQ.signal_treatment)
            print('########### - FAULT - #################')
            for i in range(len(EQ.vertices)):
                temp = ["%.2f" % i for i in EQ.vertices[i]]
                print('vertice ' + str(i) + ' = ',temp)
            print('Hypocenter location = ',geometry['hypocenter'])
            temp = ["%.2f" % i for i in EQ.Hypocenter]
            print('Hypocenter = ',temp)
            boolean = Methods.check_if_in_plane(EQ.vertices,EQ.Hypocenter,print_info=True)
            temp = ["%.2f" % i for i in EQ.site_coord]
            print('Site = ', temp)
            print('R_hyp = ', str("%.2f" % round(self.R_hypocentral,2)),' km')
            print('R_epi = ',str("%.2f" % round(self.R_epicentral,2)),' km')
            print('R_jb = ',str("%.2f" % round(self.R_jb,2)),' km')
            print('R_rup = ',str("%.2f" % round(self.R_rup,2)),' km')
            print('########### - SUBFAULTS - #################')                  
        
        self.mesh()       
        self.aggregate_signal()
        t1 = time.time()

        t_run = t1-t0
        if EQ.print_info == True:
            print('########### - TOTAL RUN TIME - #################')
            print('time [s] = ' + str("%.2f" % round(t_run,2)))
            # - Plot the layout
            Methods.plot_layout(EQ.vertices,EQ.site_coord,EQ.Hypocenter,depth=np.array(EQ.Depth)/1000.0)  

#--------------------------------------------------------------------------------------------#       
    def mesh(self):
        """
        Point source case
        """
        if EQ.nl == 1 and EQ.nl == 1:
            EQ.N_psources = len(EQ.sub_fault_centroid)
            self.PSource = []
            self.PSource.append(psource(EQ.sub_fault_centroid[0],EQ.Mo,1,self.sub_source_trup[0]))
        

        else:
            # - Create the correlated slip distribution - #
            stochastic_field = {'slip_mean':EQ.slip_mean,'slip_cov':EQ.slip_cov,
                                'seed':EQ.random_seed,'correlation_type':EQ.slip_dist,
                                'Mw':self.Mw,'H':EQ.H}
            
            
            self.source.generate_slip(EQ.L,EQ.W,EQ.nl,EQ.nw,stochastic_field=stochastic_field)
            slip = np.flip(np.transpose(self.source.slip_dist),axis=0)

            #vectorize the matrix with the asperities
            rows, columns = slip.shape[0],slip.shape[1]
            slip_vector = []
            for i in range(rows):
                for j in range(columns):
                    slip_vector.append(slip[i][j])
            
            # - Compute sub_source seismic moment - #
            EQ.N_psources = len(EQ.sub_fault_centroid)
            temp_Mo = []
            for i in range(EQ.N_psources):
                temp_Mo.append(EQ.rigidity*self.source.sub_source_area*slip_vector[i]*10**16.0)
            Mo_scale_factor = EQ.Mo/sum(temp_Mo)
                
            # - Create point sources 
            self.PSource,slip_scaled = [],[]
            for i in range(EQ.N_psources):
                slip_i = slip_vector[i]*Mo_scale_factor
                Mo_ij = EQ.rigidity*self.source.sub_source_area*slip_i*10**16.0
                slip_scaled.append(slip_i)
                    
                N_PSource_i = psource(EQ.sub_fault_centroid[i],Mo_ij,self.sub_source_activation[i]+1,self.sub_source_trup[i])
                self.PSource.append(N_PSource_i) 
            
            # - Recover the normalized slip
            self.slip_distribution = slip*Mo_scale_factor
 #--------------------------------------------------------------------------------------------#       
    def rotate_axis(self, t_standard, radial, tangential, vertical, phi):
        m = len(t_standard)
        phi = np.array(phi)
        
        def define_components(comp, i):
            if comp == []:
                component = np.zeros(m)
            else:
                component = comp[i]
            return component
        
        a_NS_i = np.zeros((m, EQ.N_psources))
        a_EW_i = np.zeros((m, EQ.N_psources))
        a_UD_i = np.zeros((m, EQ.N_psources))
        
        for i in range(EQ.N_psources):
            r = np.full((m,),define_components(radial, i))
            t = np.full((m,),define_components(tangential, i))
            z = np.full((m,),define_components(vertical, i))

            a_NS_i[:, i] = r*np.cos(phi[i]) - t*np.sin(phi[i])
            a_EW_i[:, i] = r*np.sin(phi[i]) + t*np.cos(phi[i])
            a_UD_i[:, i] = z
        
        a_NS = np.sum(a_NS_i, axis = 1)
        a_EW = np.sum(a_EW_i, axis = 1)
        a_UD = np.sum(a_UD_i, axis = 1)
        return a_NS, a_EW, a_UD
#--------------------------------------------------------------------------------------------#       
    def stochastic_noise(self,PSource):
        atr = {}
        """
        Each finite fault point source has the same frequency noise 
        """
        if EQ.random_seed_boolean is False:
            seed = np.random.randint(0,1e8)
        else:
            seed = EQ.random_seed
        
        time,wind = Methods.window_function(EQ.window,PSource.tgm,EQ.dt)
        if EQ.correlation is True:
            fft_corr = Methods.correlated_spectrum(random_seed=None)
            Noise_corr = Methods.IFFT(fft_corr, norm='ortho')
            Noise_corr =  scipy.signal.resample(Noise_corr.real, len(time))
            windowed = wind*Noise_corr
            bcw = Methods.boxcar_window(len(windowed))
            windowed = bcw*windowed
            fft_noise,freqs_noise = Methods.FFT(windowed, EQ.dt, norm='ortho')    
            fft_noise = Methods.norm_power_spec(fft_noise, freqs_noise)
            
        else:
            wNoise = Methods.white_noise(len(time),random_seed=seed)
            windowed = wNoise*wind
            bcw = Methods.boxcar_window(len(windowed))
            windowed = bcw*windowed
            fft_noise,freqs_noise = Methods.FFT(windowed, EQ.dt, norm='ortho')    
            fft_noise = Methods.norm_power_spec(fft_noise, freqs_noise)
    
        # - Scale the base spectra to the frequency noise
        atr['P_vertical'], atr['t_acc_pv'], atr['dt'] = Methods.scale_spectrum(PSource.A_p_vertical,fft_noise,freqs_noise,time,EQ.freq_ps)
        atr['P_radial'], atr['t_acc_pr'], atr['dt'] = Methods.scale_spectrum(PSource.A_p_radial,fft_noise,freqs_noise,time,EQ.freq_ps)
        atr['SV_vertical'], atr['t_acc_svv'], atr['dt'] = Methods.scale_spectrum(PSource.A_sv_vertical,fft_noise,freqs_noise,time,EQ.freq_ps)
        atr['SV_radial'], atr['t_acc_svr'], atr['dt'] = Methods.scale_spectrum(PSource.A_sv_radial,fft_noise,freqs_noise,time,EQ.freq_ps)
        atr['SH'], atr['t_acc_sh'], atr['dt'] = Methods.scale_spectrum(PSource.A_sh,fft_noise,freqs_noise,time,EQ.freq_ps)
        
        # - Compute the time array of the waves
        atr['t_pv'] = PSource.t_rupture + PSource.to_p + atr['t_acc_pv']
        atr['t_pr'] = PSource.t_rupture + PSource.to_p + atr['t_acc_pr']
        atr['t_svv'] = PSource.t_rupture + PSource.to_s + atr['t_acc_svv']
        atr['t_svr'] = PSource.t_rupture + PSource.to_s + atr['t_acc_svr']
        atr['t_sh'] = PSource.t_rupture + PSource.to_s + atr['t_acc_sh']
        
        # - Store normalization factors
        atr['H_p_ij'] = PSource.H_p_ij
        atr['H_s_ij'] = PSource.H_s_ij
        # - Store azimuth angle
        atr['phi'] = PSource.azimuth
        # - Keep track of the maximum lag time
        atr['lag'] = PSource.t_rupture + PSource.to_s

        # - Keep track of the maximum duration
        atr['duration'] = max(max(atr['t_pv']),max(atr['t_pr']),max(atr['t_svv']),max(atr['t_svr']),max(atr['t_sh']))
        return atr
#--------------------------------------------------------------------------------------------#       
    def standarize_psource(self,p):
        standarized = {}
        standarized['phi'] = p['phi']
        
        # - Colect time histories
        standarized['P_vertical'] = Methods.standarize_time(self.t_standard, p['t_pv'], p['P_vertical'].real, p['H_p_ij'], EQ.dt, p['dt'])
        standarized['P_radial'] = Methods.standarize_time(self.t_standard, p['t_pr'], p['P_radial'].real, p['H_p_ij'], EQ.dt, p['dt'])
        standarized['SV_vertical'] = Methods.standarize_time(self.t_standard, p['t_svv'], p['SV_vertical'].real, p['H_s_ij'], EQ.dt, p['dt'])
        standarized['SV_radial'] = Methods.standarize_time(self.t_standard, p['t_svr'], p['SV_radial'].real, p['H_s_ij'], EQ.dt, p['dt']) 
        standarized['SH'] = Methods.standarize_time(self.t_standard, p['t_sh'], p['SH'].real, p['H_s_ij'], EQ.dt, p['dt'])
        return standarized
#--------------------------------------------------------------------------------------------#       
    def aggregate_signal(self):
        t0 = time.time()
        point = []
        # - Introduce the frequency noise to the base-spectra
        import multiprocessing
        pool = multiprocessing.Pool(EQ.n_processors)
        point = pool.map(self.stochastic_noise,self.PSource)
        pool.close()
        pool.join()
        
        # - Retrieve the time
        count = 0
        for p in point:
            if p['lag'] > EQ.lag_max:
                EQ.lag_max = p['lag']
            if p['duration'] > EQ.Tgm_max:
                EQ.Tgm_max = p['duration']
            count += 1
        
        n_samples = int((EQ.Tgm_max + 2.)/EQ.dt)
        tf = n_samples*EQ.dt
        self.t_standard = np.linspace(0.,tf,n_samples)

        t01 = time.time()
        ## - Create an standard time array and interpolate for all subaults
        standarized = []
        pool = multiprocessing.Pool(EQ.n_processors)
        standarized = pool.map(self.standarize_psource, point)
        pool.close()
        pool.join()
        
        self.P_vertical_i,self.P_radial_i = {},{}
        self.SV_vertical_i,self.SV_radial_i,self.SH_i = {},{},{}
        phi = []
        for i in range(len(standarized)):
            phi.append(standarized[i]['phi'])
            self.P_vertical_i[i] = standarized[i]['P_vertical']
            self.P_radial_i[i] = standarized[i]['P_radial']
            self.SV_vertical_i[i] = standarized[i]['SV_vertical']
            self.SV_radial_i[i] = standarized[i]['SV_radial']
            self.SH_i[i] = standarized[i]['SH']

        t02 = time.time()
        t_run = t02-t01

        ## - Rotate axis to NS, EW, UD and aggregate individual point source contribution
        # - Time histories
        self.P_NS, self.P_EW, self.P_UD = self.rotate_axis(self.t_standard, self.P_radial_i, [], self.P_vertical_i, phi)
        self.SV_NS, self.SV_EW, self.SV_UD = self.rotate_axis(self.t_standard, self.SV_radial_i, [], self.SV_vertical_i, phi)
        self.SH_NS, self.SH_EW, self.SH_UD = self.rotate_axis(self.t_standard, [], self.SH_i, [], phi)

        ## - Join components   
        # - Time histories
        self.NS = self.P_NS + self.SV_NS + self.SH_NS
        self.EW = self.P_EW + self.SV_EW + self.SH_EW
        self.UD = self.P_UD + self.SV_UD + self.SH_UD
        t1 = time.time()
        t_run = t1-t0

        ## - Signal processing
        t0 = time.time()
        if EQ.signal_treatment is True:
            if EQ.print_info is True:
                print('Initializing signal processing')
            self.NS = Methods.SignalPostProcess(self.NS,EQ.dt,fc=EQ.fc_baseline,order=EQ.order_PostProcess)
            self.EW = Methods.SignalPostProcess(self.EW,EQ.dt,fc=EQ.fc_baseline,order=EQ.order_PostProcess)
            self.UD = Methods.SignalPostProcess(self.UD,EQ.dt,fc=EQ.fc_baseline,order=EQ.order_PostProcess)
        t1 = time.time()
        t_run = t1-t0

        
        def integration(acc):
            import scipy.integrate as it
            acc = np.array(acc)     
            vel = it.cumtrapz(acc,dx = EQ.dt,initial=0.0)
            disp = it.cumtrapz(vel,dx = EQ.dt,initial=0.0)
            return vel,disp
        
        # - Displacement and velocity time histories
        self.NS_vel,self.NS_disp = integration(self.NS)
        self.EW_vel,self.EW_disp = integration(self.EW)
        self.UD_vel,self.UD_disp = integration(self.UD)
        
        # - Time arrays
        self.t_NS = np.linspace(0.,len(self.NS)*EQ.dt, len(self.NS))
        self.t_EW = np.linspace(0.,len(self.EW)*EQ.dt, len(self.EW))
        self.t_UD = np.linspace(0.,len(self.UD)*EQ.dt, len(self.UD))
        
        # - Plots        
        if EQ.plots_boolean is True:
            t0 = time.time()
            # - NS time history
            plt.ioff()
            fig = plt.figure()
            plt.title('NS - Accelerogram',figure=fig)
            plt.plot(self.t_NS,self.NS,figure=fig)
            plt.xlabel('t [s]',fontsize = 12,figure=fig)
            plt.ylabel('acc [cm/s' +'$^2$' +']',fontsize = 12,figure=fig)
            plt.grid(figure=fig)
            plt.close(fig)
            self.NS_accelerogram = fig

            # - EW time history
            plt.ioff()
            fig = plt.figure()        
            plt.title('EW - Accelerogram',figure=fig)
            plt.plot(self.t_EW,self.EW,figure=fig)
            plt.xlabel('t [s]',fontsize = 12,figure=fig)
            plt.ylabel('acc [cm/s' +'$^2$' +']',fontsize = 12,figure=fig)
            plt.grid(figure=fig)
            plt.close(fig)
            self.EW_accelerogram = fig

            # - UD time history
            plt.ioff()
            fig = plt.figure()
            plt.title('UD - Accelerogram',figure=fig)
            plt.plot(self.t_UD,self.UD,figure=fig)
            plt.xlabel('t [s]',fontsize = 12,figure=fig)
            plt.ylabel('acc [cm/s' +'$^2$' +']',fontsize = 12,figure=fig)
            plt.grid(figure=fig)
            plt.close(fig)
            self.UD_accelerogram = fig    

            # - Acceleration histories
            plt.ioff()
            fig, (ax1, ax2,ax3) = plt.subplots(1,3, sharex='col', sharey='row')
            ax1.plot(self.t_NS,self.NS)
            ax1.set_title('NS')
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel('acc [cm/s' +'$^2$' +']')
            ax1.grid()
            ax2.plot(self.t_EW,self.EW)
            ax2.set_title('EW')
            ax2.set_xlabel('t [s]')
            ax2.grid()
            ax3.plot(self.t_UD,self.UD)
            ax3.set_title('UD')
            ax3.set_xlabel('t [s]')
            ax3.grid()
            plt.close(fig)
            self.acc_histories = fig
            
            # - Velocity histories
            plt.ioff()
            fig, (ax1, ax2,ax3) = plt.subplots(1,3, sharex='col', sharey='row')
            ax1.plot(self.t_NS,self.NS_vel)
            ax1.set_title('NS')
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel('vel [cm/s]')
            ax1.grid()
            ax2.plot(self.t_EW,self.EW_vel)
            ax2.set_title('EW')
            ax2.set_xlabel('t [s]')
            ax2.grid()
            ax3.plot(self.t_UD,self.UD_vel)
            ax3.set_title('UD')
            ax3.set_xlabel('t [s]')
            ax3.grid()
            plt.close(fig)
            self.vel_histories = fig        

            # - Displacement histories
            plt.ioff()
            fig, (ax1, ax2,ax3) = plt.subplots(1,3, sharex='col', sharey='row')
            ax1.plot(self.t_NS,self.NS_disp)
            ax1.set_title('NS')
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel('disp [cm]')
            ax1.grid()
            ax2.plot(self.t_EW,self.EW_disp)
            ax2.set_title('EW')
            ax2.set_xlabel('t [s]')
            ax2.grid()
            ax3.plot(self.t_UD,self.UD_disp)
            ax3.set_title('UD')
            ax3.set_xlabel('t [s]')
            ax3.grid()
            plt.close(fig)
            self.disp_histories = fig
            t1 = time.time()
            t_run = t1-t0
            print('time plots [s] = ' + str("%.2f" % round(t_run,2)))

 ####################-----------------------############################      
 ####################------- POINT SOURCE --------##########################  
 ####################-----------------------############################  
 
class psource(EQ):
    def __init__(self,centroid,Mo_ij,activation,t_rupture):
        self.activation = activation
        self.centroid = centroid
        self.Mo_ij = Mo_ij
        self.t_rupture = t_rupture
        self.R_hyp = np.linalg.norm(np.array(centroid) - np.array(EQ.site_coord))
        
        # - Define the soil layers
        if EQ.Depth is None:
            self.depth_list = np.linspace(0.0,self.centroid[2],2)
            self.vs = np.full((len(self.depth_list), ), EQ.Betha)
            self.vp = np.array([vs/EQ.velocity_ratio for vs in self.vs])
        else:
            # - Check if the depth array starts with zero
            self.depth_list = np.array(EQ.Depth)
            self.vs = np.array(EQ.vs)
            if EQ.vp is None:
                self.vp = [vs/EQ.velocity_ratio for vs in temp_vs]
            else:
                self.vp = np.array(EQ.vp)
        
        self.rho = np.array([Methods.density_profile(v) for v in self.vs])
        
        # - Adjust arrays for ray propagation
        if self.depth_list[0] != 0.:
            self.depth_list = np.insert(self.depth_list, 0, 0.)
            self.vs = np.insert(self.vs, 0, self.vs[0])
            self.vp = np.insert(self.vp, 0, self.vp[0])
        if self.depth_list[-1] != self.centroid[2]:
            self.depth_list = np.insert(self.depth_list, len(self.depth_list), self.centroid[2])
            self.vs = np.insert(self.vs, 0, self.vs[-1])
            self.vp = np.insert(self.vp, 0, self.vp[-1])
        
        # - S-wave propagation
        self.theta_s, self.R_s,  self.points_s, self.to_s = self.ray_propagation(self.depth_list, self.vs)
        # - P-wave propagation
        self.theta_p, self.R_p,  self.points_p, self.to_p = self.ray_propagation(self.depth_list, self.vp)
        
        self.spectra()
        t0 = time.time()
        self.A_p_vertical = self.build_spectra(self.C_p_vertical,self.fc_p_ij,'P',1.0)
        self.A_p_radial = self.build_spectra(self.C_p_radial,self.fc_p_ij,'P',1.0)
        self.A_sv_vertical = self.build_spectra(self.C_sv_vertical,self.fc_s_ij,'S',1.0)
        self.A_sv_radial = self.build_spectra(self.C_sv_radial,self.fc_s_ij,'S',1.0)
        self.A_sh = self.build_spectra(self.C_sh,self.fc_s_ij,'S',1.0)
        t1 = time.time()

        # - Duration
        if EQ.boolean_tgm == False:
            ## - Path duration
            def path_duration(R,t_R,limit):
                for lim,t in zip(limit,t_R):
                    if R <= lim:
                        subsource_duration = t(R)
                        break
                return subsource_duration
            self.t_path =  path_duration(self.R_s,EQ.path_duration,EQ.path_duration_lim)
            ## - Source duration
            self.t_source_s = EQ.source_duration(self.fc_s_ij)
            self.tgm =  self.t_path + self.t_source_s

        else:
            self.tgm =  EQ.Tgm

        # - Print info regarding point source
        if EQ.print_info == True:
            print('Point source  = ',self.activation)
            temp = ["%.2f" % i for i in self.centroid]
            print('Centroid  = ',temp)
            plane_boolean = Methods.check_if_in_plane(EQ.vertices,self.centroid,print_info=True)
            temp  = "%.2f" % self.R_hyp
            print('Hypocentral distance  = ', temp)
            temp  = "%.2f" % self.t_rupture
            print('Time to rupture  = ', temp)
            temp  = "%.2f" % self.to_s
            print('S-wave travel time  = ', temp)
            temp  = "%.2f" % self.to_p
            print('P-wave travel time  = ', temp)
            
            temp  = "%.3f" % self.fc_s_ij
            print('S-wave corner frequency   = ', temp)
            temp  = "%.3f" % self.fc_p_ij
            print('P-wave corner frequency   = ', temp)
            print('# ----------------------------#')

 #--------------------------------------------------------------------------------------------#       
    def ray_propagation(self, depth_list, velocity):
        t2 = time.time()
        
        depth_inversed = list(reversed(depth_list))
        velocity_inversed = list(reversed(velocity))
        
        XY = np.array([EQ.site_coord[0] - self.centroid[0], EQ.site_coord[1] - self.centroid[1]])
        XY_norm = np.linalg.norm(XY)
        
        ########################################
        def ray_propagation(theta,solver=True):
            Z = depth_inversed[0]
            R, points = 0., []
            p_i = np.array([self.centroid[0], self.centroid[1]])
            index = 0
            t_prop = 0.
            points.append(p_i)
            while Z > 0.:
                index += 1
                dz = Z - depth_inversed[index]
                # - Geometry
                xy = dz*np.tan(theta)
                r = dz/np.cos(theta)
                p_f = XY/XY_norm*xy + p_i

                # - Update variables for next layer
                Z -= dz
                theta = np.arcsin(velocity_inversed[index]*np.sin(theta)/velocity_inversed[index-1])
                p_i = p_f
                R += r
                t_prop += r/velocity_inversed[index-1]
                
                # - Save partial results
                points.append(p_f)
            
            error = (EQ.site_coord[0] - p_f[0]) + (EQ.site_coord[1] - p_f[1])
            if solver is True:
                return error
            else:
                return theta, R, points, t_prop  
        ########################################
        
        # - Find optimal solution
        sol = least_squares(ray_propagation, 1, bounds=((0.01,np.pi)))
        if sol.success is True:
            incidence, R,  points, t_prop  = ray_propagation(sol.x[0],solver=False)
            self.azimuth = np.arccos(XY[0]/XY_norm)
            t3 = time.time()
            t_run = t3-t2
            return incidence, R,  points, t_prop
        else:
            print('Ray propagation did not SUCCEED')
        
#--------------------------------------------------------------------------------------------#            
    def spectra(self):
        # - Compute the radiation patterns
        dip = EQ.dip*np.pi/180.0
        rake = EQ.rake*np.pi/180.0
        self.RP_p = np.sqrt(4.0/15.0)
        self.RP_sv = 0.5*np.sqrt(np.sin(rake)**2.0*(14.0/15.0 + 1.0/3.0*np.sin(2.0*dip)**2.0) +
                                  np.cos(rake)**2.0*(4.0/15.0 + 2.0/3.0*np.cos(dip)**2.0))
        self.RP_sh = 0.5*np.sqrt(2.0/3.0*np.cos(rake)**2.0*(1+np.sin(dip)**2.0) +
                                  1.0/3.0*np.sin(rake)**2.0*(1+np.cos(2.0*dip)**2.0))
        
        # - Compute the free surface factors
        FS_sv_radial,FS_sv_vertical,FS_p_radial,FS_p_vertical  = self.free_surface_calculator(EQ.Betha, EQ.Alpha, self.theta_s, self.theta_p)
        FS_sh = 2.0
        
        # - Energy partition
        EP_p_radial = -1*np.sin(self.theta_p)
        EP_p_vertical = np.cos(self.theta_p)
        EP_sv_radial = EP_p_vertical
        EP_sv_vertical = np.sin(self.theta_s)
        EP_sh = 1.0
        
        # - Corner frequencies
        ## - subfault
        self.fc_s_ij = 4.9e6*EQ.Betha*(EQ.stress_drop/(EQ.Mo*min(self.activation/float(EQ.N_psources),EQ.F_pulse/100.0)))**(1.0/3.0)
        self.fc_p_ij = EQ.Alpha*self.fc_s_ij/EQ.Betha
        ## - Fault
        self.fc_s = 4.9e6*EQ.Betha*(EQ.stress_drop/EQ.Mo*EQ.F_pulse/100.0)**(1.0/3.0)
        self.fc_p = EQ.Alpha*self.fc_s/EQ.Betha
        
        # - Scaling factors Otarola
        def scale_ratio(f,fc,fcij):
            global_f = 0.0
            local_f = 0.0
            for i in range(len(f)):
                global_f += (f[i]**2/(1+(f[i]/fc)**2))**2
                local_f += (f[i]**2/(1+(f[i]/fcij)**2))**2
            return global_f,local_f
        [global_p,local_p] = scale_ratio(EQ.freq_ps,self.fc_p,self.fc_p_ij) 
        self.H_p_ij = EQ.Mo/self.Mo_ij*np.sqrt(1.0/EQ.N_psources*(global_p/local_p))
        [global_s,local_s] = scale_ratio(EQ.freq_ps,self.fc_s,self.fc_s_ij) 
        self.H_s_ij = EQ.Mo/self.Mo_ij*np.sqrt(1.0/EQ.N_psources*(global_s/local_s))
        
        # - Spectrum constant
        self.C_p_vertical = self.RP_p*FS_p_vertical*EP_p_vertical*self.Mo_ij/(4*np.pi*EQ.Rho*EQ.Alpha**3)*(10**-20)
        self.C_p_radial = self.RP_p*FS_p_radial*EP_p_radial*self.Mo_ij/(4*np.pi*EQ.Rho*EQ.Alpha**3)*(10**-20)
        self.C_sv_vertical = self.RP_sv*FS_sv_vertical*EP_sv_vertical*self.Mo_ij/(4*np.pi*EQ.Rho*EQ.Betha**3)*(10**-20)
        self.C_sv_radial = self.RP_sv*FS_sv_radial*EP_sv_radial*self.Mo_ij/(4*np.pi*EQ.Rho*EQ.Betha**3)*(10**-20)
        self.C_sh = self.RP_sh*FS_sh*EP_sh*self.Mo_ij/(4*np.pi*EQ.Rho*EQ.Betha**3)*(10**-20)
                    
 #--------------------------------------------------------------------------------------------#          
    def free_surface(self):
        discr = 20
        ### - According to Evans 1984 and Otarola 2018
        crit = np.arcsin(EQ.Betha/EQ.Alpha)
        # - Secondary wave
        j = np.concatenate((np.linspace(0.0,crit,discr),np.linspace(crit*1.05,np.pi/2,discr)))
        n = len(j)
        i,p,X,Y,Z = np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
        SV_radial,SV_vertical = np.zeros(n),np.zeros(n)
        
        for index in range(n):
            if j[index] <= crit:
                i[index] = np.arcsin(np.sin(j[index])*EQ.Alpha/EQ.Betha)
            p[index] = np.sin(j[index])/EQ.Betha
            X[index] = 1/EQ.Betha**2 - 2*p[index]**2
            Y[index] = 4*np.cos(i[index])*np.cos(j[index])/(EQ.Alpha*EQ.Betha)
            Z[index] = X[index]**2 + p[index]**2*Y[index]
            SV_radial[index] = abs(2*X[index]/(EQ.Betha**2*Z[index]))
            SV_vertical[index] = Y[index]/(EQ.Betha**2*Z[index])
            
        # - Primary wave
        i = np.linspace(0.0,np.pi/2,discr)
        n = len(i)
        j_p,p,X,Y,Z = np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
        P_radial,P_vertical = np.zeros(n),np.zeros(n)
        for index in range(n):
            j_p[index] = np.arcsin(np.sin(i[index])/EQ.Alpha*EQ.Betha)
            p[index] = np.sin(j_p[index])/EQ.Betha
            X[index] = 1/EQ.Betha**2 - 2*p[index]**2
            Y[index] = 4*np.cos(i[index])*np.cos(j_p[index])/(EQ.Alpha*EQ.Betha)
            Z[index] = X[index]**2 + p[index]**2*Y[index]
            P_vertical[index] = abs(2*X[index]/(EQ.Betha**2*Z[index]))
            P_radial[index] = Y[index]/(EQ.Betha**2*Z[index])
        
        # - Plot
        if EQ.plots_boolean == True:
            plt.ioff()
            fig = plt.figure()
            plt.plot(j,SV_radial,'y-',figure=fig)
            plt.plot(j,SV_vertical,'b-',figure=fig)
            plt.plot(i,P_radial,'r-',figure=fig)
            plt.plot(i,P_vertical,'g-',figure=fig)
            plt.xlabel('Incidence angle [rad]',figure=fig)
            plt.ylabel('Free surface factor [ ]',figure=fig)
            plt.grid(figure=fig)
            plt.close(fig)
            self.free_surface_plot = fig
        return j,SV_radial,SV_vertical,i,P_vertical,P_radial 
 #--------------------------------------------------------------------------------------------#          
    def free_surface_calculator(self,betha, alpha, incidence_s, incidence_p):
        ### - According to Evans 1984 and Otarola 2018
        crit = np.arcsin(betha/alpha)
        # - Secondary wave
        j = incidence_s 
        if j <= crit:
            i = np.arcsin(np.sin(j)*alpha/betha)
        else:
            i = 0.0
        p = np.sin(j)/betha
        X = 1.0/betha**2 - 2*p**2
        Y = 4.0*np.cos(i)*np.cos(j)/(alpha*betha)
        Z = X**2.0 + p**2.0*Y
        SV_radial = abs(2*X/(betha**2*Z))
        SV_vertical = Y/(betha**2.0*Z)

        # - Primary wave
        i = incidence_p
        j_p = np.arcsin(np.sin(i)/alpha*betha)
        p = np.sin(j_p)/betha
        X = 1/betha**2 - 2*p**2
        Y = 4*np.cos(i)*np.cos(j_p)/(alpha*betha)
        Z = X**2 + p**2*Y
        P_vertical = abs(2*X/(betha**2*Z))
        P_radial = Y/(betha**2*Z)

        return SV_radial,SV_vertical,P_radial,P_vertical
 #--------------------------------------------------------------------------------------------#       
    def build_spectra(self,C,fc,wave_type,factor):
        if factor == None:
            f = 1.0
        else:
            f = factor
        
        self.spec_shape = 1./(1. + (EQ.freq_ps/fc)**2.)
        
        if wave_type is 'S':
            self.P = self.path(wave_type, self.R_s)
        else:
            self.P = self.path(wave_type, self.R_p)
        
        self.G = Methods.site(EQ.amp_type, EQ.amp_freq, EQ.amp_funct,
                              EQ.Hypocenter, self.depth_list, self.vs,
                              self.rho, EQ.Betha, EQ.Rho, EQ.f_max,
                              EQ.kappa, EQ.freq_ps, kappa_type=EQ.kappa_type, 
                              shift_freq=EQ.shift_freq)
        
        self.I = (2.*m.pi*EQ.freq_ps)**2.
        A = C*self.spec_shape*self.P*self.G*self.I*f
        return A
 #--------------------------------------------------------------------------------------------#       
    def path(self,wave_type, distance):
        ### --- Paramters: R - Distance from source to site --- ###
        ## - Asuming a cq value = betha
        # - Geometrical Spreading
        def geom_spreading(R):
            if EQ.b2 is None:
                gs = R**EQ.b1
            else:
                if R <= EQ.geom_spread_lim:
                    gs = R**EQ.b1
                else:
                    gs = R**EQ.b2
            return gs
        
        self.Z = geom_spreading(distance)
        
        # - Quality factor (Aki,1980)
        if wave_type == 'P':
                factor = 3*EQ.Alpha**2/(4*EQ.Betha**2)
                v = EQ.Alpha
        else:
                factor = 1.0
                v = EQ.Betha
                
        Q = np.array([max(EQ.Qo,EQ.Q1*(fi)**EQ.Q1exp) for fi in EQ.freq_ps])
        P = self.Z*np.exp(-1.*m.pi*EQ.freq_ps*distance/(Q*v))
        return P
