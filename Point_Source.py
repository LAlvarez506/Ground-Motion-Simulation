import numpy as np
import scipy
from scipy.interpolate import interp1d
import multiprocessing as mp
from Methods import GMSM
import random
import math as m
import time

Methods = GMSM()

"""
Method that constructs the FAS of a point source.  The spectra are computed for P, SV and SH waves.
The input is given as a list, so as to facilitate the use of multiprocessing.
Entry parameters:
    - X[0]: Coordinates of the point source in local axis, i.e. not geographic coordinates [lon, lat, depth]
    - X[1]: Siesmic moment associated to the point source, in dyne-cm
    - X[2]: Activation order with respect to all other point sources within a finite fault model
    - X[3]: time for the rupture to arrive from the hypocenter to the point source, in seconds
    - X[4]: Coordinates of the site of observation in local axis
    
    - X[5]: Simulation settings, dictionary
        
"""

class point_source():
    #--------------------------------------------------------------------------------------------#    
    ### - RUN THE COMPUTATIONS - ###
    def __init__(self, sett):
        
        # --------------------------------------------------- #
        ## - ENTRY DATA - ##
        # - Source parameters
        self.Mo_ij, self.activation, t_rupture = sett['Mo_ij'], sett['activation'], sett['t_rupture']

        # - Coordinates
        ps_centroid, site_coord = sett['ps_centroid'], sett['site_coord']
        self.dip, self.rake = sett['dip'], sett['rake']
        
        # - Propagation and site
        depth_medium, vs_medium, vp_medium, rho_medium = sett['depth'], sett['vs'], sett['vp'], sett['rho']
        velocity_ratio = sett['velocity_ratio']
        self.TF = sett['TF']
        
        # - Duration
        boolean_tgm, path_duration = sett['boolean_tgm'], sett['path_duration']
        boolean_tgm_source = sett['boolean_tgm_source']
        Tgm_source, Tgm = sett['Tgm_source'], sett['Tgm']
        
        # - Global parameters
        self.stress_drop, self.F_pulse = sett['stress_drop'], sett['F_pulse']
        self.Mo, self.N_psources = sett['Mo'], sett['N_psources']
        self.dt, self.fc_p, self.fc_s = sett['dt'], sett['fc_p'], sett['fc_s']
        self.freq_ps = sett['freq_ps']
        self.amp_freq, self.amp = sett['amp_freq'], sett['amp']
        self.kappa, self.f_max = sett['kappa'], sett['f_max']
        self.site_lowpass, self.f_max_exp = sett['site_lowpass'], sett['f_max_exp']
        self.R1, self.b1 = sett['R1'], sett['b1']
        self.R2, self.b2 = sett['R2'], sett['b2']
        self.Qo, self.Q1, self.Q1exp = sett['Qo'], sett['Q1'], sett['Q1exp']

        # --------------------------------------------------- #
        ### - GEOMETRICAL COMPUTATION - ###
        self.R_hyp = np.linalg.norm(np.array(ps_centroid) - np.array(site_coord))
        self.R_alternative = (self.R_hyp**2. + ps_centroid[2]**2.)**0.5
        
        dx = site_coord[0] - ps_centroid[0]
        dy = site_coord[1] - ps_centroid[1]
        theta = np.arctan(dy/dx)
        azimuth = np.pi/2. - theta if dx > 0. else 3.*np.pi/2. - theta
        
        # --------------------------------------------------- #
        ### - DIRECT RAY PROPAGATION - ###
        # - Define the soil layers
        if depth_medium is None:
            print('Must provide a 1D description of the propagation medium')

        else:
            # - Propagation lists
            if depth_medium[-1] < ps_centroid[2]:
                print('Description of the medium does not reach the source - ERROR!')
            else:
                depth_list = np.array(depth_medium)
                vs = np.array(vs_medium)
                
                if vp_medium is None:
                    vp = [vs/velocity_ratio for vs in temp_vs]
                else:
                    vp = np.array(vp_medium)
                    
                if rho_medium is None:
                    rho = Methods.density_profile(vs)
                else:
                    rho = np.array(rho_medium)

                
                # - Characteristics at the source
                index_source = np.where(depth_list >= ps_centroid[2])[0][0]
                self.betha = vs[index_source-1]
                self.alpha = vp[index_source-1]
                self.rho_s = rho[index_source-1]
                
                # - Define the propagation arrays
                depth_list = depth_list[0:index_source]
                self.vs = vs[0:index_source]
                self.vp = vp[0:index_source] 
                self.rho = rho[0:index_source] 
                
                self.z = np.insert(depth_list, len(depth_list), ps_centroid[2]) 

        # - S-wave propagation
        data_s = {'z':self.z, 'v':self.vs, 'start_point':ps_centroid, 'target':site_coord}
        R_s, self.theta_s, to_s = Methods.ray_propagation_bisection(1e-3, np.pi/2., 100, 0.01, data_s)

        # - P-wave propagation
        data_p = {'z':self.z, 'v':self.vp, 'start_point':ps_centroid, 'target':site_coord}
        R_p, self.theta_p, to_p = Methods.ray_propagation_bisection(1e-3, np.pi/2., 100, 0.01, data_p)
        self.R_s, self.R_p = self.R_hyp, self.R_hyp

        # --------------------------------------------------- #
        ### - UNDERLINE SPECTRA - ###
        self.spectra()
        t0 = time.time()
        A_p_vertical = self.build_spectra(self.C_p_vertical, self.fc_p_ij, 'P', 1., polarization='vertical')
        A_p_radial = self.build_spectra(self.C_p_radial,self.fc_p_ij,'P',1., polarization='radial')
        A_sv_vertical = self.build_spectra(self.C_sv_vertical,self.fc_s_ij,'SV',1., polarization='vertical')
        A_sv_radial = self.build_spectra(self.C_sv_radial,self.fc_s_ij,'SV',1., polarization='radial')
        A_sh = self.build_spectra(self.C_sh,self.fc_s_ij,'SH',1., polarization='tangential')
        
        # --------------------------------------------------- #
        ### - SIGNAL DURATION - ###
        # - Duration
        if boolean_tgm == False:
            ## - Path duration
            t_path_s =  path_duration*self.R_s
            t_path_p =  path_duration*self.R_p
            ## - Source duration
            if boolean_tgm_source is False:
                t_source_s = 1./self.fc_s_ij
                t_source_p = 1./self.fc_p_ij
            else:
                t_source_s = Tgm_source
                t_source_p = Tgm_source
            ## - Total duration
            tgm_s =  t_path_s + t_source_s
            tgm_p =  t_path_p + t_source_p
        else:
            tgm_s =  Tgm
            tgm_p =  Tgm
            
        self.point_source_output = {'A_p_vertical':A_p_vertical, 'A_p_radial':A_p_radial, 'A_sv_vertical':A_sv_vertical,
                               'A_sv_radial':A_sv_radial, 'A_sh':A_sh, 'to_p':to_p, 'to_s':to_s, 'H_p_ij':self.H_p_ij,
                               'H_s_ij':self.H_s_ij, 'azimuth':azimuth, 'incidence_p':self.theta_p, 'incidence_s':self.theta_s,
                               'tgm_p':tgm_p, 'tgm_s':tgm_s, 't_rupture':t_rupture}

#--------------------------------------------------------------------------------------------#            
    ### - OUTPUT - ###
    def ps_info(self):
        return [self.point_source_output]
    
#--------------------------------------------------------------------------------------------#            
    ### - CONSTANTS AND SOURCE SPECTRA - ###
    def spectra(self):
        # - Compute the radiation patterns
        dip = self.dip*np.pi/180.
        rake = self.rake*np.pi/180.
        RP_p = np.sqrt(4./15.)
        RP_sv = 0.5*np.sqrt(np.sin(rake)**2.*(14./15. + 1./3.*np.sin(2.*dip)**2.) +
                                  np.cos(rake)**2.*(4./15. + 2./3.*np.cos(dip)**2.))
        RP_sh = 0.5*np.sqrt(2./3.*np.cos(rake)**2.*(1.+ np.sin(dip)**2.) +
                                  1./3.*np.sin(rake)**2.*(1. + np.cos(2.*dip)**2.))
        
        # --------------------------------------------------- #
        # - Compute the free surface factors
        if self.TF is None:
            FS_p_vertical, FS_p_radial = Methods.FS_pwaves(self.theta_p, self.alpha, self.betha)
            FS_sv_vertical, FS_sv_radial = Methods.FS_swaves(self.theta_s, self.alpha, self.betha)
            FS_sh = 2.
            
            # - Energy partition
            EP_p_radial = -np.sin(self.theta_p)
            EP_p_vertical = np.cos(self.theta_p)
            EP_sv_radial = np.cos(self.theta_s)
            EP_sv_vertical = np.sin(self.theta_s)
            EP_sh = 1.
        else:
            FS_p_vertical, FS_p_radial, FS_sv_vertical, FS_sv_radial, FS_sh = 1., 1., 1., 1., 1.
            EP_p_radial, EP_p_vertical, EP_sv_radial, EP_sv_vertical, EP_sh = 1., 1., 1., 1., 1.
            
        # --------------------------------------------------- #
        # - Corner frequencies
        self.fc_s_ij = 4.9e6*self.betha*(self.stress_drop/(self.Mo*min(self.activation/float(self.N_psources), self.F_pulse/100.0)))**(1./3.)
        self.fc_p_ij = self.alpha*self.fc_s_ij/self.betha

        # - Scaling factors Otarola
        def scale_ratio(fc,fcij):
            frequencies = np.linspace(0.01, 0.5/self.dt, 100)
            global_f = 0.0
            local_f = 0.0
            for i in range(len(frequencies)):
                global_f += (frequencies[i]**2./(1.+(frequencies[i]/fc)**2.))**2.
                local_f += (frequencies[i]**2./(1.+(frequencies[i]/fcij)**2.))**2.
            return global_f,local_f
        
        global_p, local_p = scale_ratio(self.fc_p, self.fc_p_ij)        
        self.H_p_ij = self.Mo/self.Mo_ij*np.sqrt(1./self.N_psources*(global_p/local_p))
        
        global_s, local_s = scale_ratio(self.fc_s, self.fc_s_ij) 
        self.H_s_ij = self.Mo/self.Mo_ij*np.sqrt(1./self.N_psources*(global_s/local_s))

        # --------------------------------------------------- #
        # - Spectrum constant
        self.C_p_vertical = RP_p*FS_p_vertical*EP_p_vertical*self.Mo_ij/(4.*np.pi*self.rho_s*self.alpha**3.)*(10.**-20.)
        self.C_p_radial = RP_p*FS_p_radial*EP_p_radial*self.Mo_ij/(4.*np.pi*self.rho_s*self.alpha**3.)*(10.**-20.)
        self.C_sv_vertical = RP_sv*FS_sv_vertical*EP_sv_vertical*self.Mo_ij/(4.*np.pi*self.rho_s*self.betha**3.)*(10.**-20.)
        self.C_sv_radial = RP_sv*FS_sv_radial*EP_sv_radial*self.Mo_ij/(4.*np.pi*self.rho_s*self.betha**3.)*(10.**-20.)
        self.C_sh = RP_sh*FS_sh*EP_sh*self.Mo_ij/(4.*np.pi*self.rho_s*self.betha**3.)*(10.**-20.)

 #--------------------------------------------------------------------------------------------#     
    ### - CONSTRUCT THE SPECTRA - ###
    def build_spectra(self, C, fc, wave_type, factor, polarization=None):
        if factor == None:
            f = 1.0
        else:
            f = factor
        R = self.R_s if wave_type is 'SV' or 'SH' else self.R_p       
        P = self.path(wave_type, R)
        G = self.site()
        spec_shape = 1./(1. + (self.freq_ps/fc)**2.)
        I = (2.*m.pi*self.freq_ps)**2.
        A = C*spec_shape*P*G*I*f
        return A
 #--------------------------------------------------------------------------------------------#       
    ### - SITE FILTER - ###
    def site(self):
        # --------------------------------------------------- #
        # - Amplification
        Amp = Methods.site_amplification(self.freq_ps, self.amp_freq, self.amp)
        # --------------------------------------------------- #
        # - Attenuation
        D = Methods.site_attenuation(self.freq_ps, kappa=self.kappa, f_max=self.f_max,
                                     lowpass=self.site_lowpass, f_max_exp=self.f_max_exp)
        return Amp*D
 #--------------------------------------------------------------------------------------------#       
    ### - SITE FILTER - ###
    def path(self, wave_type, distance):
        # --------------------------------------------------- #
        # - Geometrical attenuation
        def geom_spreading(R):
            if R <= self.R1 :
                gs = R**self.b1
            elif R <= self.R2:
                gs = self.R1**self.b1
            else:
                gs = self.R1**self.b1*(self.R2/R)**self.b2
            return gs
        
        Z = geom_spreading(distance)
        
        # --------------------------------------------------- #
        # - Ana-elastic attenuation
        # - Quality factor (Aki,1980)
        if wave_type == 'P':
            factor_wave_type = 3.*(self.alpha/self.betha)**2./4.
            v = self.alpha
        else:
            factor_wave_type = 1.
            v = self.betha
        
        Q = np.array([max(self.Qo, self.Q1*fi**self.Q1exp)*factor_wave_type for fi in self.freq_ps])
        P = Z*np.exp(-1.*m.pi*self.freq_ps*distance/(Q*v))
        return P
