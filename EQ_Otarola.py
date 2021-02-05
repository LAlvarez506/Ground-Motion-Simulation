
from Point_Source import point_source as ps_generate
import numpy as np
import scipy
import multiprocessing as mp
from Methods import GMSM
from GSource import gsource
import random
import math as m
import time
Methods = GMSM()

class EQ(object):
    # - Medium
    site_coord = None
    Hypocenter = None
    Depth, vs, vp, Rho = None, None, None, None
    v_rupture = 0.80

    # - Duration
    path_duration = None
    lag_max = 0.0
    Tgm_max = 0.0
    Tgm = 0.0
    boolean_tgm = False
    boolean_tgm_source = False
    dt = 0.01
    samples_t = 0
    t = []
    freq_ps = np.linspace(0., 100., 1001)
    fc = 0.0
    
    # - Window signal
    window = {}
    window['epsilon_window'] = None
    window['nu_window'] = None
    window['f_tgm'] = None
    window['window_type'] = None

    # - Source
    ## - Geometry
    rake = None
    dip = None
    random_strike = None
    random_dip = None
    random_hypocenter = False
    
    ## - Parameters
    stress_drop = None #bar
    F_pulse = None
    velocity_ratio = None
    Mo = 0.0
    hypocenter = []
    site_coord = []
    N_psources = 0
    freq_ps = np.linspace(0., 100., 1001)
    dt = 0.01

    Qo, Q1, Q1exp = None, None, None
    R1, b1, R2, b2 = None, None, None, None
    
    # - The site
    amp_freq, amp = None, None
    kappa = None
    f_max = None
    f_max_exp = None
    site_lowpass = None
    TF = None
    
    # - Signal processing
    signal_treatment = False
    fc_baseline = None
    order_PostProcess = 3

    incidence_manual = None
    
    def __init__(self, Mw, source_file=None, parameters=None, print_info=None, n_sim=None, n_processors=None, 
                 point_source=True):

        t0 = time.time()
        self.point_source = point_source
        self.source_file = source_file
        
        if print_info is not None:
            self.print_info = print_info

        # - Parallel processing
        if n_sim is None:
            self.n_sim = 1
        else:
            self.n_sim = n_sim

        if n_processors is None:
            self.n_processors = 1
        else:
            self.n_processors = n_processors

        self.parameters = parameters
        self.Mw = Mw
        EQ.Mo = 10.**((Mw + 10.74)*3./2.)

        if 'dt' in parameters:
            if parameters['dt'] is not None:
                EQ.dt = parameters['dt']

        # ----- WINDOW FUNCTION ----- #
        if 'window_type' in parameters:
            self.window['window_type'] = parameters['window_type']
        if 'epsilon_window' in parameters:
            self.window['epsilon_window'] = parameters['epsilon_window']
        if 'nu_window' in parameters:
            EQ.window['nu_window'] = parameters['nu_window']
        if 'f_tgm' in parameters:
            EQ.window['f_tgm'] = parameters['f_tgm']
            
        # ----- DURATION ----- #
        if 'Tgm' in parameters and parameters['Tgm'] is not None:
                EQ.Tgm = parameters['Tgm']
                EQ.boolean_tgm = True
        else:
            if 'Tgm_path' in parameters:
                EQ.path_duration = parameters['Tgm_path']
            else:
                print('Path related duration not specified')
            if 'Tgm_source' in parameters:
                if parameters['Tgm_source'] is 'courboulex_sub':
                    Mo_Nm = EQ.Mo*10.**-7.
                    T = 10.**(0.28*np.log10(Mo_Nm)-4.32)
                    s_ln = 0.32
                    """
                    Obtain the underline normal distribution moments to sample
                    from the lognormal distribution
                    """
                    mu_normal = np.log(T**2./(T**2. + s_ln**2.)**0.5)
                    sigma_normal = (np.log(1. + (s_ln/T)**2.))**0.5
                    EQ.Tgm_source = np.random.lognormal(mean=mu_normal, sigma=sigma_normal)
                    EQ.boolean_tgm_source = True
                elif parameters['Tgm_source'] is 'courboulex_other':
                    Mo_Nm = EQ.Mo*10.**-7.
                    T = 10.**(0.31*np.log10(Mo_Nm)-4.90)
                    s_ln = 0.34
                    """
                    Obtain the underline normal distribution moments to sample
                    from the lognormal distribution
                    """
                    mu_normal = np.log(T**2./(T**2. + s_ln**2.)**0.5)
                    sigma_normal = (np.log(1. + (s_ln/T)**2.))**0.5
                    EQ.Tgm_source = np.random.lognormal(mean=mu_normal, sigma=sigma_normal)
                    EQ.boolean_tgm_source = True
                else:
                    pass
                    # - Takes the inverse of the corner frequency 
            else:
                EQ.Tgm_source = None
        # -------------------------------------------------------- #
        # ----- SOURCE ----- #
        if 'rake' in parameters:
            EQ.rake = parameters['rake']
        else:
            print('Rake not included in parameters')
 
        if 'strike' in parameters:
            if 'random_strike' in parameters:
                if parameters['random_strike'] is True:
                    EQ.strike = random.uniform(0.0001,360.)
                
            elif parameters['strike'] == 0.0:
                EQ.strike = 0.0001
            else:
                EQ.strike = parameters['strike']
                
        if 'dip' in parameters:
            if 'random_dip' in parameters:
                if parameters['random_dip'] is True:
                    EQ.dip = random.uniform(0.0001,90.) 
            elif parameters['dip'] == 0.0:
                EQ.dip = 0.0001
            else:
                EQ.dip = parameters['dip']

        if 'random_hypocenter' in self.parameters:
           EQ.random_hypocenter = self.parameters['random_hypocenter']
        
        if 'hypocenter' in self.parameters:
           self.hypocenter = self.parameters['hypocenter']   
        
        if 'F_pulse' in parameters:
            if parameters['F_pulse'] is not None:
                EQ.F_pulse = parameters['F_pulse']
        if 'stress_drop' in parameters:
            if parameters['stress_drop'] is not None:
                EQ.stress_drop = parameters['stress_drop']
        
        # --------------------------------------------------- #
        # - Description of the propagation medium
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
        if 'vp' in parameters:
            """
            Velocity profile (P-waves) for the soil layer depth array, in m/2
            """
            if parameters['vp'] is not None:
                EQ.vp = parameters['vp'] 
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
                
        # - Check the propagation medium arrays
        if EQ.Depth[0] != 0.:
            if len(EQ.Depth) != len(EQ.vs)-1:
                print('When not includig the surface in the depth array, the velocity array must have one more element than the depth')
            else:
                EQ.Depth = np.insert(EQ.Depth, 0, 0.)
        else:
            if len(EQ.Depth) != len(EQ.vs):
                print('When includig the surface in the depth array, the depth must have the same lenght as the velocity array')       
                
        # --------------------------------------------------- #
        # - Path related parameters
        if 'R1' in parameters:
            if parameters['R1'] is not None:
                EQ.R1 = parameters['R1']
        if 'b1' in parameters:
            if parameters['b1'] is not None:
                EQ.b1 = parameters['b1']  
        if 'R2' in parameters:
            if parameters['R2'] is not None:
                EQ.R2 = parameters['R2']
        if 'b2' in parameters:
            if parameters['b2'] is not None:
                EQ.b2 = parameters['b2']
        if 'Qo' in parameters:
            if parameters['Qo'] is not None:
                EQ.Qo = parameters['Qo']
        if 'Q1' in parameters:
            if parameters['Q1'] is not None:
                EQ.Q1 = parameters['Q1']
        if 'Q1exp' in parameters:
            if parameters['Q1exp'] is not None:
                EQ.Q1exp = parameters['Q1exp']
        # -------------------------------------------------------- #     
        # - Site related parameters
        if 'kappa' in parameters:
            if parameters['kappa'] is not None:
                EQ.kappa = parameters['kappa']
        if 'f_max' in parameters:
            if parameters['f_max'] is not None:
                EQ.f_max = parameters['f_max']
        if 'f_max_exp' in parameters:
            if parameters['f_max_exp'] is not None:
                EQ.f_max_exp = parameters['f_max_exp']
        if 'amp' in parameters:
            if parameters['amp'] is not None:
                EQ.amp = parameters['amp']
        if 'amp_freq' in parameters:
            if parameters['amp_freq'] is not None:
                EQ.amp_freq = parameters['amp_freq']
        # - Including site effects by means of transfer functions
        if 'TF' in parameters:
            if parameters['TF'] is not None:
                EQ.TF = parameters['TF']
        if 'incidence_manual' in parameters:
            if parameters['incidence_manual'] is not None:
                EQ.incidence_manual = parameters['incidence_manual']
        # -------------------------------------------------------- #        
        # ----- SIGNAL PROCESSING ----- #        
        if 'signal_treatment' in parameters:
            if parameters['signal_treatment'] is not None:
                EQ.signal_treatment = parameters['signal_treatment']
        if 'fc_baseline' in parameters:
            if parameters['fc_baseline'] is not None:
                EQ.fc_baseline = parameters['fc_baseline'] 
        
        # --------------------------------------------------- #
        # - Seed
        if 'seed' in parameters:
                if parameters['seed'] is not None:
                    np.random.seed(parameters['seed'])
                else:
                    seed = np.random.randint(0,10000.)
                    np.random.seed(seed)
        # --------------------------------------------------- #
        # - Common parameters
        self.params = {'dt':EQ.dt, 'sim_seed':[], 'PSource':None,
                  'standard_freqs':EQ.freq_ps,'signal_treatment':EQ.signal_treatment, 'TF':EQ.TF,
                  'order_PostProcess':EQ.order_PostProcess,'fc_baseline':EQ.fc_baseline, 'incidence_manual':EQ.incidence_manual}
        
        # --------------------------------------------------- #
        # - Conduct simulation
        self.run_simulation()
                    
 ######################################################################
    ## - CONDUCT SIMULATIONS - ##
    def run_simulation(self):
        
        # --------------------------------------------------- #
        # - Construct the ruptures
        sett = {'Mo_ij':EQ.Mo, 'activation':1, 'site_coord':None,
                'ps_centroid':None, 'dip':EQ.dip, 'rake':EQ.rake,
                'depth':EQ.Depth, 'vs':EQ.vs, 'vp':EQ.vp, 'rho':EQ.Rho,
                'velocity_ratio':EQ.velocity_ratio, 'TF':EQ.TF, 't_rupture':0.,
                'boolean_tgm':EQ.boolean_tgm, 'path_duration':EQ.path_duration,
                'boolean_tgm_source':EQ.boolean_tgm_source, 'Tgm_source':EQ.Tgm_source,
                'Tgm':EQ.Tgm, 'stress_drop':EQ.stress_drop, 'F_pulse':EQ.F_pulse,
                'Mo':EQ.Mo, 'N_psources':1, 'dt':EQ.dt, 'fc_p':None, 'fc_s':None,
                'Qo':EQ.Qo, 'Q1':EQ.Q1, 'Q1exp':EQ.Q1exp, 'freq_ps':EQ.freq_ps, 
                'R1':EQ.R1, 'b1':EQ.b1, 'R2':EQ.R2, 'b2':EQ.b2, 'amp_freq':EQ.amp_freq, 
                'amp':EQ.amp, 'kappa':EQ.kappa, 'f_max':EQ.f_max, 'site_lowpass':EQ.site_lowpass,
                'f_max_exp':EQ.f_max_exp}
        
        if self.point_source is True:
            
            EQ.Hypocenter = [0., 0., self.hypocenter[2]]
            EQ.ps_centroid = EQ.Hypocenter
            EQ.site_coord = Methods.locate_site(self.hypocenter, self.parameters['site'], reference=EQ.Hypocenter, depth=True)
            
            self.side_computations()
            sett['ps_centroid'], sett['site_coord'] = EQ.ps_centroid, EQ.site_coord
            sett['fc_p'], sett['fc_s'] = EQ.fc_p, EQ.fc_s
            # ----------------------------
            # - point source construction
            Y = ps_generate(sett)
            psource = Y.ps_info()
            N_psources = 1
            
        else:
            # ----------------------------
            # - Geometry
            Y = self.set_geometry(self.source_file)
            EQ.Hypocenter, EQ.site_coord, EQ.ps_centroid = Y[0], Y[1], Y[2]
            ff_mo, ff_activation, ff_rupture_time = Y[3], Y[4], Y[5]
            # ----------------------------
            # - Side computations
            self.side_computations()
            sett['site_coord'] = EQ.site_coord
            sett['fc_p'], sett['fc_s'] = EQ.fc_p, EQ.fc_s
            
            # ----------------------------
            # - point source construction
            psource = self.mesh(EQ.ps_centroid, ff_mo, ff_activation, ff_rupture_time, sett)
            N_psources = len(psource)
            
        # --------------------------------------------------- #
        # - Run the simulation
        self.sim_acc = []
        self.theta = []
        for i in range(self.n_sim):
            p = self.params.copy()
            seeds = [np.random.randint(1e8) for k in range(N_psources)]
            
            # --------------------------------------------------- #
            # - Generate the windows
            Windows = {}
            duration = 0.
            for i in range(N_psources):
                time_p, wind_p = Methods.window_function(EQ.window, psource[i]['tgm_p'], EQ.dt)
                time_s, wind_s = Methods.window_function(EQ.window, psource[i]['tgm_s'], EQ.dt)
                Windows[i] = {'P':{'time':time_p, 'wind':wind_p}, 'S':{'time':time_s, 'wind':wind_s}}
                total_time = max(psource[i]['to_s'] + psource[i]['t_rupture'] + max(time_s), psource[i]['to_p'] + psource[i]['t_rupture'] + max(time_p))
                
                if total_time > duration:
                    duration = total_time
            duration = round(duration, 0) + 1.
            n_samples = int(duration/EQ.dt)
            
            # --------------------------------------------------- #
            # - Save point source information
            PointSources = []
            for i in range(N_psources):
                ps = {}
                ps['A_p_vertical'] = psource[i]['A_p_vertical']
                ps['A_p_radial'] = psource[i]['A_p_radial']
                ps['A_sv_vertical'] = psource[i]['A_sv_vertical']
                ps['A_sv_radial'] = psource[i]['A_sv_radial']
                ps['A_sh'] = psource[i]['A_sh']
                ps['window'] = Windows[i]
                ps['to_p'] = psource[i]['to_p']
                ps['to_s'] = psource[i]['to_s']
                ps['t_rupture'] = psource[i]['t_rupture']
                ps['n_samples'] = n_samples
                ps['H_p_ij'] = psource[i]['H_p_ij']
                ps['H_s_ij'] = psource[i]['H_s_ij']
                ps['phi'] = psource[i]['azimuth']
                ps['incidence_s'] = psource[i]['incidence_s']
                ps['incidence_p'] = psource[i]['incidence_p']
                ps['order_PostProcess'] = EQ.order_PostProcess
                PointSources.append(ps)
            
            p['PSource'] = PointSources
            p['sim_seed'] = seeds
            self.theta.append(p)
            
        accelerations = self.run_multiple_simulation()
        self.sim_acc += accelerations

  ######################################################################
    def set_geometry(self, source_file):

        # --------------------------------------------------- #
        # - Import rupture information
        self.source = gsource(source_file)
            
        L, W = self.source.x.max(), self.source.z.max()
        ff_subsources_centroid = self.source.sub_sources_centroid
        ff_slip = self.source.slip
        ff_mo = EQ.Mo*ff_slip/ff_slip.sum()
        ff_rupture_time = self.source.rupt_t
        ff_activation = self.source.activation
        
        self.hyp_local = np.array(self.source.hypocenter) 
        
        # --------------------------------------------------- # 
        # - Rotate the fault to align with dip and strike
        # - Dip
        sub_sources_centroid_dip = self.source.rotate_plane(ff_subsources_centroid, EQ.dip, 'x')
        hyp_dip = self.source.rotate_plane(self.hyp_local, EQ.dip, 'x')

        # - Strike
        sub_sources_centroid_aligned = self.source.rotate_plane(sub_sources_centroid_dip, EQ.strike, 'z')
        hyp_aligned = self.source.rotate_plane(hyp_dip, EQ.strike, 'z')
        
        # - Update the Z coordinate based on the real hypocenter depth
        dz = self.parameters['hypocenter'][2] - hyp_aligned[2]
        Hypocenter = [hyp_aligned[0] - hyp_aligned[0], hyp_aligned[1]- hyp_aligned[1], hyp_aligned[2]+dz]
        site_coord = Methods.locate_site(self.parameters['hypocenter'], self.parameters['site'], reference=Hypocenter, depth=True)                        
        sub_fault_centroid = [[p[0]- hyp_aligned[0], p[1]- hyp_aligned[1], p[2]+dz] for p in sub_sources_centroid_aligned]
        
        return [Hypocenter, site_coord, sub_fault_centroid, ff_mo, ff_activation, ff_rupture_time]

        
  ######################################################################
    ## - AID COMPUTATIONS - ##
    def side_computations(self):
        # --------------------------------------------------- #
        # - Medium parameters at the hypocenter depth
        index_depth = np.where(EQ.Depth >= EQ.Hypocenter[2])[0][0]
        Betha_hyp = EQ.vs[index_depth-1]
        
        if EQ.vp is not None:
            Alpha_hyp = EQ.vp[index_depth-1]
        else:
            Alpha_hyp = Betha_hyp/EQ.velocity_ratio

        # --------------------------------------------------- #
        # - Compute corner frequencies
        EQ.fc_s = 4.9e6*Betha_hyp*(EQ.stress_drop/(EQ.Mo))**(1./3.)
        EQ.fc_p = Alpha_hyp*EQ.fc_s/Betha_hyp

        # --------------------------------------------------- #
        # - Compute distances
        # - Get hypocentral and epicentral distances
        self.R_hypocentral = np.linalg.norm(np.array(EQ.site_coord) - np.array(EQ.Hypocenter))
        self.R_epicentral = np.linalg.norm(np.array(EQ.site_coord) - np.array([EQ.Hypocenter[0],EQ.Hypocenter[1],0.0]))
        self.R_rup, self.R_jb, self.R_x = Methods.rupture_distances(EQ.ps_centroid, EQ.site_coord)
        
        # --------------------------------------------------- #
        # - Consider a station located underground
        if EQ.site_coord[2] > 0.:
            index_depth = np.where(EQ.Depth >= EQ.site_coord[2])[0][0]
            EQ.Depth, EQ.vs, EQ.vp = EQ.Depth[index_depth-1:], EQ.vs[index_depth-1:], EQ.vp[index_depth-1:]
            EQ.Depth[0] = EQ.site_coord[2]
 
 ######################################################################
    def mesh(self, ps_centroid, ff_mo, ff_activation, ff_rupture_time, sett):
        # --------------------------------------------------- #
        # - Update the number of sources
        N_psources = sum(x > 0 for x in ff_mo)
        sett['N_psources'] = N_psources

        # --------------------------------------------------- #
        # - Create the point sources
        X = []
        for i in range(ff_mo.shape[0]):
            if ff_mo[i] > 0.:
                sett_ps = sett.copy()
                sett_ps['Mo_ij'], sett_ps['activation'] = ff_mo[i], ff_activation[i]
                sett_ps['ps_centroid'], sett_ps['t_rupture'] = ps_centroid[i], ff_rupture_time[i]
                X.append(sett_ps)
                
        
        N_psources = len(X)
        # - Create point sources 
        PSource = []
        pool = mp.Pool(self.n_processors)
        PSource = pool.map(ps_generate, X)
        pool.close()
        pool.join()
        psource_info = [p.ps_info()[0] for p in PSource]
        return psource_info
        
 ########################################################################
    def run_multiple_simulation(self):
        p = mp.Pool(self.n_processors)
        points = p.map(add_noise,self.theta)
        p.close()
        p.join()
        
        p1 = mp.Pool(self.n_processors)
        results = p1.map(aggregate_signal,points)
        p1.close()
        p1.join()
        
        return results      
    
############-------------------------------------###############
############---- SIGNAL CONSTRUCTION  ---- ###############
############-------------------------------------###############
"""
Programming based focused on multiprocessing
"""

######################################################################    
### - CONVOLUTION OF NOISE AND UNDERLINE SPECTRA - ###
def add_noise(params):
    # --------------------------------------------------- #
    # - Entry parameters
    PSource = params['PSource']
    sim_seed = params['sim_seed']
    dt = params['dt']
    standard_freqs = params['standard_freqs']
    signal_treatment = params['signal_treatment']
    order_PostProcess = params['order_PostProcess']
    fc_baseline = params['fc_baseline']
    TF = params['TF']
    incidence_manual = params['incidence_manual']

    # --------------------------------------------------- #
    # - Generate the modulated noise 
    point = []
    N_psources = len(PSource)
    incidence_p, incidence_s = [], []
    # - Introduce the frequency noise to the base-spectra
    for i in range(N_psources):
        atr = {}
        time = np.arange(0., PSource[i]['n_samples']*dt, dt)
        
        # --------------------------------------------------- #
        ## - Create the windows
        # - P wave
        wn = PSource[i]['window']['P']['wind']
        start_index =  int((PSource[i]['to_p'] + PSource[i]['t_rupture'])/dt)
        P_window = np.concatenate((np.zeros(start_index), wn, np.zeros(PSource[i]['n_samples'] - (len(wn) + start_index))))
        
        # - S wave
        wn = PSource[i]['window']['S']['wind']
        start_index =  int((PSource[i]['to_s'] + PSource[i]['t_rupture'])/dt)
        S_window = np.concatenate((np.zeros(start_index), wn, np.zeros(PSource[i]['n_samples'] - (len(wn) + start_index))))

        """
        Each finite fault point source has the same frequency noise 
        """
        # --------------------------------------------------- #
        ## - Produce the white noise
        seed = sim_seed[i]
        wNoise = Methods.white_noise(len(time),random_seed=seed)
        
        # - P waves
        windowed_p = wNoise*P_window
        bcw = Methods.boxcar_window(len(windowed_p))
        windowed_p = bcw*windowed_p
        fft_noise_p, freqs_noise_p = Methods.FFT(windowed_p, EQ.dt, norm='ortho')    
        fft_noise_p = Methods.norm_power_spec(fft_noise_p, freqs_noise_p)
        
        # - S waves
        windowed_s = wNoise*S_window
        bcw = Methods.boxcar_window(len(windowed_s))
        windowed_s = bcw*windowed_s
        fft_noise_s, freqs_noise_s = Methods.FFT(windowed_s, EQ.dt, norm='ortho')    
        fft_noise_s = Methods.norm_power_spec(fft_noise_s, freqs_noise_s)

        incidence_p.append(PSource[i]['incidence_p'])
        incidence_s.append(PSource[i]['incidence_s'])
        
        # - Incidence angles
        if incidence_manual is None:
            theta_p, theta_s = PSource[i]['incidence_p'], PSource[i]['incidence_s']
        else:
            theta_p, theta_s = incidence_manual, incidence_manual
        
        # --------------------------------------------------- #
        # - Convolve noise and spectra
        atr['P_vertical'],atr['t_acc_pv'] = Methods.scale_spectrum(PSource[i]['A_p_vertical'], fft_noise_p, freqs_noise_p, time, standard_freqs,
                                                                   factor=PSource[i]['H_p_ij'], TF=TF, incidence=theta_p,
                                                                   polarization='vertical', wave_type='P')
        atr['P_radial'],atr['t_acc_pr'] = Methods.scale_spectrum(PSource[i]['A_p_radial'], fft_noise_p, freqs_noise_p, time, standard_freqs,
                                                                 factor=PSource[i]['H_p_ij'], TF=TF, incidence=theta_p,
                                                                   polarization='radial', wave_type='P')
        atr['SV_vertical'],atr['t_acc_svv'] = Methods.scale_spectrum(PSource[i]['A_sv_vertical'], fft_noise_s, freqs_noise_s, time, standard_freqs,
                                                                     factor=PSource[i]['H_s_ij'], TF=TF, incidence=theta_s,
                                                                   polarization='vertical', wave_type='SV')
        atr['SV_radial'],atr['t_acc_svr'] = Methods.scale_spectrum(PSource[i]['A_sv_radial'], fft_noise_s, freqs_noise_s, time, standard_freqs,
                                                                   factor=PSource[i]['H_s_ij'], TF=TF, incidence=theta_s,
                                                                   polarization='radial', wave_type='SV')
        atr['SH'],atr['t_acc_sh'] = Methods.scale_spectrum(PSource[i]['A_sh'],fft_noise_s, freqs_noise_s, time, standard_freqs, 
                                                           factor=PSource[i]['H_s_ij'], TF=TF, incidence=theta_s,
                                                                   polarization='tangential', wave_type='SH')

        atr['phi'] = PSource[i]['phi']
        atr['signal_treatment'] = signal_treatment
        atr['order_PostProcess'] = order_PostProcess
        atr['fc_baseline'] = fc_baseline
        atr['dt'] = dt
        atr['n_samples'] = PSource[i]['n_samples']
        point.append(atr)

    return point
######################################################################  
### - AGGREGATION OF THE SIGNALS - ###
def aggregate_signal(point):
    # --------------------------------------------------- #
    # - Entry parameters
    dt = point[0]['dt']
    N_psources = len(point)
    signal_treatment = point[0]['signal_treatment']
    order_PostProcess = point[0]['order_PostProcess']
    fc_baseline = point[0]['fc_baseline']
    n_samples = point[0]['n_samples']
    
    # --------------------------------------------------- #
    # - Rotate signals to orthogonal components
    t_standard = np.linspace(0., dt*n_samples, n_samples)
    NS, EW, UD = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)
    for i in range(N_psources):
        # - Azimuth angle
        phi = point[i]['phi']
        
        # - Rotation matrix
        M_rot = np.array([[np.cos(phi), -np.sin(phi), 0.], [np.sin(phi), np.cos(phi), 0.], [0., 0., 1.]])
        
        # - P waves
        P_NS = point[i]['P_radial']*np.cos(phi)
        P_EW = point[i]['P_radial']*np.sin(phi)
        P_UD = point[i]['P_vertical']
        
        # - SV waves
        SV_NS = point[i]['SV_radial']*np.cos(phi)
        SV_EW = point[i]['SV_radial']*np.sin(phi)
        SV_UD = point[i]['SV_vertical']
        
        # - SH waves
        SH_NS = point[i]['SH']*-np.sin(phi)
        SH_EW = point[i]['SH']*np.cos(phi)
        
        NS += P_NS + SV_NS + SH_NS 
        EW += P_EW + SV_EW + SH_EW
        UD += P_UD + SV_UD 
        
    # --------------------------------------------------- #
    # - Process signal
    if signal_treatment is True:
        NS = Methods.SignalPostProcess(NS,dt,fc=fc_baseline,order=order_PostProcess)
        EW = Methods.SignalPostProcess(EW,dt,fc=fc_baseline,order=order_PostProcess)
        UD = Methods.SignalPostProcess(UD,dt,fc=fc_baseline,order=order_PostProcess)

    return NS,EW,UD
