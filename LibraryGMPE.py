import numpy as np
import math 

def Zhao_2006(T, M, x, h, Vs30, FR, SI, SS, MS):
    """
    
    % by James Bronder 08/11/2010
    % Stanford University
    % jbronder@stanford.edu
    
    % Purpose: Computes the median and dispersion of the Peak Ground
    % Acceleration, PGA, or the Spectral Acceleration, Sa, at 5% Damping.
    
    % Citation: "Attentuation Relations of Strong Ground Motion in Japan Using
    % Site Classification Based on Predominant Period" by Zhao, John X., Jian
    % Zhang, Akihiro Asano, Yuki Ohno, Taishi Oouchi, Toshimasa Takahashi,
    % Hiroshi Ogawa, Kojiro Irikura, Hong K. Thio, Paul G. Somerville, Yasuhiro
    % Fukushima, and Yoshimitsu Fukushima. Bulletin of the Seismological
    % Society of America, Vol. 96, No. 3, pp. 898-913, June 2006
    
    % General Limitations: This particular model utilizes data primarily from
    % earthquakes from Japan, with some data from the western United States.
    % The model accounts for shallow crustal events (including reverse fault
    % type mechanisms), and subduction zone events, both interface and in-slab
    % (intraslab) types. The model yields results for focal depths no greater
    % than 120 km. The focal depth, h, is the greatest influence in this model.
    
    %-------------------------------INPUTS------------------------------------%
    
    % T  = Period (sec)
    % M  = Moment Magnitude
    % x  = Source to Site distance (km); Defines as the shortest distance from
    %      the site to the rupture zone. NOTE: 'x' must be a positive,
    %      non-negative value.
    % h  = Focal (Hypocentral) Depth (km)
    % Vs30  = Average Shear Velocity in the first 30 meters of the soil profile
    %     (m/sec)
    % FR = Reverse-Fault Parameter: FR = 1, For Crustal Earthquakes ONLY IF a
    %                                       Reverse-Fault Exists
    %                               FR = 0, Otherwise
    % SI = Source-Type Indicator: SI = 1, For Interface Events
    %                             SI = 0, Otherwise
    % SS = Source-Type Indicator: SS = 1, For Subduction Slab Events
    %                             SS = 0, Otherwise
    % MS = Magnitude-Squared Term: MS = 1, Includes the Magnitude-squared term
    %                              MS = 0, Does not include magnitude-squared
    %                                      term
    
    %------------------------------OUTPUTS------------------------------------%
    
    % Sa    = Median spectral acceleration prediction (g)
    % sigma = Logarithmic standard deviation of spectral acceleration
    %         prediction
    """
    
    #-------------------------------Period------------------------------------%
    
    period = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.25, 1.5, 2., 2.5, 3., 4., 5.])
    
    #-------------------General Equation Coefficients-------------------------%
    a = [1.101,  1.076,  1.118,  1.134,  1.147,  1.149,  1.163,  1.2,  1.25,  1.293,  1.336,  1.386,  1.433,  1.479,  1.551,  1.621,  1.694,  1.748,  1.759,  1.826,  1.825]
    b = [-0.00564,  -0.00671,  -0.00787,  -0.00722,  -0.00659,  -0.0059,  -0.0052,  -0.00422,  -0.00338,  -0.00282,  -0.00258,  -0.00242,  -0.00232,  -0.0022,  -0.00207,  -0.00224,  -0.00201,  -0.00187,  -0.00147,  -0.00195,  -0.00237]
    c = [0.0055,  0.0075,  0.009,  0.01,  0.012,  0.014,  0.015,  0.01,  0.006,  0.003,  0.0025,  0.0022,  0.002,  0.002,  0.002,  0.002,  0.0025,  0.0028,  0.0032,  0.004,  0.005]
    d = [1.08, 1.06, 1.083, 1.053, 1.014, 0.966, 0.934, 0.959, 1.008, 1.088, 1.084, 1.088, 1.109, 1.115, 1.083, 1.091, 1.055, 1.052, 1.025, 1.044, 1.065]
    e = [0.01412, 0.01463, 0.01423, 0.01509, 0.01462, 0.01459, 0.01458, 0.01257, 0.01114, 0.01019, 0.00979, 0.00944, 0.00972, 0.01005, 0.01003, 0.00928, 0.00833, 0.00776, 0.00644, 0.0059, 0.0051]
    Fr = [0.251, 0.251, 0.24, 0.251, 0.26, 0.269, 0.259, 0.248, 0.247, 0.233, 0.22, 0.232, 0.22, 0.211, 0.251, 0.248, 0.263, 0.262, 0.307, 0.353, 0.248]
    Si = [0, 0, 0, 0, 0, 0, 0, -0.041, -0.053, -0.103, -0.146, -0.164, -0.206, -0.239, -0.256, -0.306, -0.321, -0.337, -0.331, -0.39, -0.498]
    Ss = [2.607, 2.764, 2.156, 2.161, 1.901, 1.814, 2.181, 2.432, 2.629, 2.702, 2.654, 2.48, 2.332, 2.233, 2.029, 1.589, 0.966, 0.789, 1.037, 0.561, 0.225]
    Ssl = [-0.528, -0.551, -0.42, -0.431, -0.372, -0.36, -0.45, -0.506, -0.554, -0.575, -0.572, -0.54, -0.522, -0.509, -0.469, -0.379, -0.248, -0.221, -0.263, -0.169, -0.12]
    
    #------------Site Class Coefficients & Prediction Uncertainty-------------%
    CH = [0.293, 0.939, 1.499, 1.462, 1.28, 1.121, 0.852, 0.365, -0.207, -0.705, -1.144, -1.609, -2.023, -2.451, -3.243, -3.888, -4.783, -5.444, -5.839, -6.598, -6.752]
    C1 = [1.111, 1.684, 2.061, 1.916, 1.669, 1.468, 1.172, 0.655, 0.071, -0.429, -0.866, -1.325, -1.732, -2.152, -2.923, -3.548, -4.41, -5.049, -5.431, -6.181, -6.347]
    C2 = [1.344, 1.793, 2.135, 2.168, 2.085, 1.942, 1.683, 1.127, 0.515, -0.003, -0.449, -0.928, -1.349, -1.776, -2.542, -3.169, -4.039, -4.698, -5.089, -5.882, -6.051]
    C3 = [1.355, 1.747, 2.031, 2.052, 2.001, 1.941, 1.808, 1.482, 0.934, 0.394, -0.111, -0.62, -1.066, -1.523, -2.327, -2.979, -3.871, -4.496, -4.893, -5.698, -5.873]
    C4 = [1.42, 1.814, 2.082, 2.113, 2.03, 1.937, 1.77, 1.397, 0.955, 0.559, 0.188, -0.246, -0.643, -1.084, -1.936, -2.661, -3.64, -4.341, -4.758, -5.588, -5.798]
    s_t = [0.604, 0.64, 0.694, 0.702, 0.692, 0.682, 0.67, 0.659, 0.653, 0.653, 0.652, 0.647, 0.653, 0.657, 0.66, 0.664, 0.669, 0.671, 0.667, 0.647, 0.643]
    t_t = [0.398, 0.444, 0.49, 0.46, 0.423, 0.391, 0.379, 0.39, 0.389, 0.401, 0.408, 0.418, 0.411, 0.41, 0.402, 0.408, 0.414, 0.411, 0.396, 0.382, 0.377]
    S_T = [0.723, 0.779, 0.849, 0.839, 0.811, 0.786, 0.77, 0.766, 0.76, 0.766, 0.769, 0.77, 0.771, 0.775, 0.773, 0.779, 0.787, 0.786, 0.776, 0.751, 0.745]
    
    #----------Magnitude Squared Interevent Uncertainty Coefficents-----------%
    Qc = [0, 0, 0, 0, 0, 0, 0, 0, -0.0126, -0.0329, -0.0501, -0.065, -0.0781, -0.0899, -0.1148, -0.1351, -0.1672, -0.1921, -0.2124, -0.2445, -0.2694]
    Wc = [0, 0, 0, 0, 0, 0, 0, 0, 0.0116, 0.0202, 0.0274, 0.0336, 0.0391, 0.044, 0.0545, 0.063, 0.0764, 0.0869, 0.0954, 0.1088, 0.1193]
    t_c = [0.303, 0.326, 0.342, 0.331, 0.312, 0.298, 0.3, 0.346, 0.338, 0.349, 0.351, 0.356, 0.348, 0.338, 0.313, 0.306, 0.283, 0.287, 0.278, 0.273, 0.275]
    QI = [0, 0, 0, -0.0138, -0.0256, -0.0348, -0.0423, -0.0541, -0.0632, -0.0707, -0.0771, -0.0825, -0.0874, -0.0917, -0.1009, -0.1083, -0.1202, -0.1293, -0.1368, -0.1486, -0.1578]
    WI = [0, 0, 0, 0.0286, 0.0352, 0.0403, 0.0445, 0.0511, 0.0562, 0.0604, 0.0639, 0.067, 0.0697, 0.0721, 0.0772, 0.0814, 0.088, 0.0931, 0.0972, 0.1038, 0.109]
    t_I = [0.308, 0.343, 0.403, 0.367, 0.328, 0.289, 0.28, 0.271, 0.277, 0.296, 0.313, 0.329, 0.324, 0.328, 0.339, 0.352, 0.36, 0.356, 0.338, 0.307, 0.272]
    Ps = [0.1392, 0.1636, 0.169, 0.1669, 0.1631, 0.1588, 0.1544, 0.146, 0.1381, 0.1307, 0.1239, 0.1176, 0.1116, 0.106, 0.0933, 0.0821, 0.0628, 0.0465, 0.0322, 0.0083, -0.0117]
    Qs = [0.1584, 0.1932, 0.2057, 0.1984, 0.1856, 0.1714, 0.1573, 0.1309, 0.1078, 0.0878, 0.0705, 0.0556, 0.0426, 0.0314, 0.0093, -0.0062, -0.0235, -0.0287, -0.0261, -0.0065, 0.0246]
    Ws = [-0.0529, -0.0841, -0.0877, -0.0773, -0.0644, -0.0515, -0.0395, -0.0183, -0.0008, 0.0136, 0.0254, 0.0352, 0.0432, 0.0498, 0.0612, 0.0674, 0.0692, 0.0622, 0.0496, 0.015, -0.0268]
    t_s = [0.321, 0.378, 0.42, 0.372, 0.324, 0.294, 0.284, 0.278, 0.272, 0.285, 0.29, 0.299, 0.289, 0.286, 0.277, 0.282, 0.3, 0.292, 0.274, 0.281, 0.296]
    
    # - Preliminary Computations & Constants
    if h >= 125.:
        h = 125.
    elif h < 125.:
        h
    
    hc = 15
    if h >= hc:
        delh = 1
    elif h < hc:
        delh = 0
    
    # -  Ground Motion Prediction Computation
    if T == 1000:
        Sa, sigma = np.zeros(period.shape[0]), np.zeros(period.shape[0])
        for i in range(period.shape[0]):
            sa_temp, sigma_temp = Zhao_2006(period[i],M,x,h,Vs30,FR,SI,SS,MS)
            Sa[i], sigma[i] = sa_temp, sigma_temp
        
        return period, Sa, sigma
    
    elif T not in period:
        
        index_lower = np.where(period < T)[0][-1]
        index_higher = np.where(period > T)[0][0]
        T_lo = period[index_lower]
        T_hi = period[index_higher]

        Sa_lo, sigma_lo = Zhao_2006(T_lo,M,x,h,Vs30,FR,SI,SS,MS)
        Sa_hi, sigma_hi = Zhao_2006(T_hi,M,x,h,Vs30,FR,SI,SS,MS)
        
        X = [T_lo,T_hi]
        Y_Sa = [Sa_lo, Sa_hi]
        Y_sigma = [sigma_lo, sigma_hi]
        Sa = np.interp(T,X,Y_Sa)
        sigma = np.interp(T,X,Y_sigma)
        
        return Sa, sigma
        
    else:
        i = np.where(period==T)[0][0]
        
        if MS == 1:
            if SS == 1:
                Mc = 6.5
                Pst = Ps[i]
                Qst = Qs[i]
                Wst = Ws[i]
                t_t = t_s[i]
                lnS_MS = Pst*(M - Mc) + Qst*((M - Mc)**2.) + Wst
            elif FR == 1:
                Mc = 6.3
                Pst = 0.0
                Qst = Qc[i]
                Wst = Wc[i]
                t_t = t_c[i
]
                lnS_MS = Pst*(M - Mc) + Qst*((M - Mc)**2.) + Wst
            elif SI == 1:
                Mc = 6.3
                Pst = 0.0
                Qst = QI[i]
                Wst = WI[i]
                t_t = t_I[i]
                lnS_MS = Pst*(M - Mc) + Qst*(M - Mc)**2. + Wst
            else:
                Mc = 6.3
                Pst = 0.0
                Qst = Qc[i]
                Wst = Wc[i]
                t_t = t_c[i]
                lnS_MS = Pst*(M - Mc) + Qst*((M - Mc)**2.) + Wst
                
        elif MS == 0:
            t_t = t_t[i]
            lnS_MS = 0


        if  FR == 1:
            FR = Fr[i]
            SI = 0
            SS = 0
            SSL = 0
        elif SI == 1:
            FR = 0
            SI = Si[i]
            SS = 0
            SSL = 0
        elif SS == 1:
            FR = 0
            SI = 0
            SS = Ss[i]
            SSL = Ssl[i]
        else:
            FR = 0
            SI = 0
            SS = 0
            SSL = 0
        
        if Vs30 <= 200:
            Ck = C4[i]
        elif Vs30 >= 200 and Vs30 <= 300:
            Ck = C3[i]
        elif Vs30 > 300 and Vs30 <= 600:
            Ck = C2[i]
        elif Vs30 > 600 and Vs30 < 1100:
            Ck = C1[i]
        elif Vs30 >= 1100:
            Ck = CH[i]
        
        r = x + c[i]*np.exp(d[i]*M)
        
        lnY = a[i]*M + b[i]*x - np.log(r) + e[i]*(h-hc)*delh + FR + SI + SS + SSL*np.log(x) + Ck + lnS_MS; 
        
        sigma = (s_t[i]**2. + t_t**2.)**0.5
        
        Sa = np.exp(lnY)/981.; #Median Sa in g
        return Sa, sigma

########################################################################################################################
def Idriss_2014(T, M, Rrup, Vs30, F):
    """
    ############################################################################
    # INPUT
    #
    #   M               = moment magnitude
    #   Vs30(m/sec)     = average shear wave velocity over the top 30m below
    #                     the ground surface
    #   T               = period of vibration
    #                     Use 1000 for output the array of Sa with period
    #   R_RUP           = closest distance to fault rupture (rupture distance)(km)
    #   F               = 0 strike slip & other non-reverse faulting
    #                   = 1 reverse
    #
    # OUTPUT   
    #
    #   Sa              = median spectral acceleration prediction
    #   sigma           = logarithmic standard deviation of spectral acceleration
    ############################################################################
    """
    #---------------------------------------------------------#
    def I_2014_sub(M, ip, R_RUP, Vs30,F):
        period = [0., 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,	0.4, 0.5, 0.75, 1., 1.5, 2., 3., 4., 5., 7.5, 10.]
        if M <= 6.75:
            a1 = [7.0887, 7.1157, 7.2087, 6.2638, 5.9051, 7.5791, 8.0190, 9.2812, 9.5804, 9.8912, 9.5342, 9.2142, 8.3517,	7.0453,	5.1307,	3.3610,	0.1784,	-2.4301, -4.3570, -7.8275, -9.2857]
            a2 = [0.2058, 0.2058, 0.2058, 0.0625, 0.1128, 0.0848, 0.1713, 0.1041, 0.0875, 0.0003, 0.0027, 0.0399, 0.0689, 0.1600, 0.2429, 0.3966, 0.7560, 0.9283, 1.1209, 1.4016, 1.5574]
            b1 = [2.9935, 2.9935, 2.9935, 2.8664, 2.9406, 3.0190, 2.7871, 2.8611, 2.8289, 2.8423, 2.8300, 2.8560, 2.7544, 2.7339, 2.6800, 2.6837, 2.6907, 2.5782, 2.5468, 2.4478, 2.3922]
            b2 = [-0.2287, -0.2287, -0.2287, -0.2418, -0.2513, -0.2516, -0.2236, -0.2229, -0.2200, -0.2284, -0.2318, -0.2337, -0.2392, -0.2398, -0.2417, -0.2450, -0.2389, -0.2514, -0.2541, -0.2593, -0.2586]
        else:
            a1 = [9.0138, 9.0408, 9.1338, 7.9837, 7.7560, 9.4252, 9.6242, 11.1300, 11.3629, 11.7818, 11.6097, 11.4484, 10.9065, 9.8565, 8.3363, 6.8656, 4.1178, 1.8102,0.0977, -3.0563, -4.4387]
            a2 = [-0.0794, -0.0794, -0.0794, -0.1923, -0.1614, -0.1887, -0.0665, -0.1698, -0.1766, -0.2798, -0.3048, -0.2911, -0.3097, -0.2565, -0.2320, -0.1226, 0.1724, 0.3001, 0.4609, 0.6948, 0.8393]
            b1 = [2.9935, 2.9935, 2.9935, 2.7995, 2.8143, 2.8131, 2.4091, 2.4938, 2.3773, 2.3772, 2.3413, 2.3477, 2.2042, 2.1493, 2.0408, 2.0013, 1.9408, 1.7763, 1.7030, 1.5212, 1.4195]
            b2 = [-0.2287, -0.2287, -0.2287, -0.2319, -0.2326, -0.2211, -0.1676, -0.1685, -0.1531, -0.1595, -0.1594, -0.1584, -0.1577, -0.1532, -0.1470, -0.1439, -0.1278, -0.1326, -0.1291, -0.1220, -0.1145]
        
        a3 = [0.0589,  0.0589,  0.0589,  0.0417,  0.0527,  0.0442,  0.0329,  0.0188,  0.0095,  -0.0039,  -0.0133,  -0.0224,  -0.0267,  -0.0198,  -0.0367,  -0.0291,  -0.0214,  -0.024,  -0.0202,  -0.0219,  -0.0035]
        zeta = [-0.854, -0.854, -0.854, -0.631, -0.591, -0.757, -0.911, -0.998, -1.042, -1.030, -1.019, -1.023, -1.056, -1.009, -0.898, -0.851, -0.761, -0.675, -0.629, -0.531, -0.586]
        gamma = [-0.0027, -0.0027, -0.0027, -0.0061, -0.0056, -0.0042, -0.0046, -0.0030, -0.0028, -0.0029, -0.0028, -0.0021, -0.0029, -0.0032, -0.0033, -0.0032, -0.0031, -0.0051, -0.0059, -0.0057, -0.0061]
        phi = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.06, 0.04, 0.02, 0.02, 0, 0, 0, 0]
        
        ln_PSA = a1[ip] + a2[ip] * M + a3[ip]*(8.5 - M)**2. - (b1[ip] + b2[ip] * M) * np.log(R_RUP + 10) + zeta[ip] * np.log(Vs30) + gamma[ip] * R_RUP + phi[ip]*F
        Sa = np.exp(ln_PSA)
        
        M_SE = max(min(M, 7.5), 5)
        
        if period[ip] < 0.05:
            sigma = 1.18 + 0.035 * np.log(0.05) - 0.06 * M_SE
        elif period[ip] > 3.:
            sigma = 1.18 + 0.035 * np.log(3.) - 0.06 * M_SE
        else:
            sigma = 1.18 + 0.035 * np.log(period[ip]) - 0.06 * M_SE
        
        return Sa, sigma
    #--------------------------------------------------------#
    
    # for the given period T, get the index for the constants
    period = np.array([0., 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1., 1.5, 2., 3., 4., 5., 7.5, 10.])
    
    
    def interp_PSA(Ti, index_Tlower, index_Thigher):
        T_higher = period[index_Thigher]
        Sa_higher, sigma_higher = I_2014_sub(M, index_Thigher, Rrup, Vs30, F)

        T_lower = period[index_Tlower]
        Sa_lower, sigma_lower = I_2014_sub(M, index_Tlower, Rrup, Vs30, F)
        
        T_array = [T_lower, T_higher]
        Sa_array = [np.log(Sa_lower), np.log(Sa_higher)]
        sigma_array = [sigma_lower, sigma_higher]
        Sa = np.exp(np.interp(Ti, T_array, Sa_array))
        sigma = np.interp(Ti, T_array, sigma_array)
        return Sa, sigma
                
    
    if Vs30 > 1200.:
        Vs30 = 1200.

    if (T == 1000):
        n_periods_construct = len(period)
        Sa = np.zeros(n_periods_construct)
        sigma = np.zeros(n_periods_construct)
        for i in range(n_periods_construct):
            Sa[i], sigma[i] = I_2014_sub(M, i, Rrup, Vs30, F)
        return Sa, sigma, period
    
    elif (type(T) == list or type(T) == np.ndarray):
        n_T = len(T)
        Sa = np.zeros(n_T)
        sigma = np.zeros(n_T)
        for i in range(n_T):
            Ti = T[i]
            if Ti not in period:
                # - Get the indexes of the periods below and above
                ## - Higher
                index_higher = np.where(period > Ti)[0][0]
                index_lower = np.where(period > Ti)[0][-1]
                Sa[i], sigma[i] = interp_PSA(Ti, index_lower, index_higher)
            else:
                index = np.where(period == Ti)[0][0]
                Sa[i], sigma[i] = I_2014_sub(M, index, Rrup, Vs30, F)

        return Sa, sigma
    else:
        if T == 0.:
            T = min(period)
        if T not in period:
            # - Get the indexes of the periods below and above
            ## - Higher
            index_higher = np.where(period > T)[0][0]
            index_lower = np.where(period < T)[0][-1]
            Sa, sigma = interp_PSA(T, index_lower, index_higher)
        else:
            index = np.where(period == T)[0][0]
            Sa, sigma = I_2014_sub(M, index, Rrup, Vs30, F)

    return Sa, sigma
        
########################################################################################################################
def BSSA_2014_nga(M, T, Rjb, Fault_Type, region, z1, Vs30, full_std=False):

# BSSA14 NGA-West2 model, based on teh following citation
#
# Boore, D. M., Stewart, J. P., Seyhan, E., and Atkinson, G. M. (2014). 
# “NGA-West2 Equations for Predicting PGA, PGV, and 5# Damped PSA for 
# Shallow Crustal Earthquakes.” Earthquake Spectra, 30(3), 1057–1085.
# http://www.earthquakespectra.org/doi/abs/10.1193/070113EQS184M
#
# Provides ground-mtion prediction equations for computing medians and
# standard devisations of average horizontal component internsity measures
# (IMs) for shallow crustal earthquakes in active tectonic regions.
###########################################################################
# Input Variables
# M = Moment Magnitude
# T = Period (sec); Use Period = -1 for PGV computation
#                 Use 1000 to output the array of median with original period
#                 (no interpolation)
# Rjb = Joyner-Boore distance (km)
# Fault_Type    = 0 for unspecified fault
#               = 1 for strike-slip fault
#               = 2 for normal fault
#               = 3 for reverse fault
# region        = 0 for global (incl. Taiwan)
#               = 1 for California
#               = 2 for Japan
#               = 3 for China or Turkey
#               = 4 for Italy
# z1            = Basin depth (km); depth from the groundsurface to the
#                   1km/s shear-wave horizon.
#               = 999 if unknown
# Vs30          = shear wave velocity averaged over top 30 m in m/s
#
# Output Variables
# median        = Median amplitude prediction
# sigma         = NATURAL LOG standard deviation 
###########################################################################
    period = np.array([0., 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1., 1.5, 2., 3., 4., 5., 7.5, 10.])
    n_period = len(period)
    
    U, SS, NS, RS = 0, 0, 0, 0
    if Fault_Type == 0:
        U = 1.
    elif Fault_Type == 1:
        SS = 1
    elif Fault_Type == 2:
        NS = 1
    elif Fault_Type == 3:
        RS = 1
        
    U = (Fault_Type == 0)
    SS = (Fault_Type == 1)
    NS = (Fault_Type == 2)
    RS = (Fault_Type == 3)
    
    
    if type(T) is int and T == 1000:
        Sa = []
        sigma = []
        for ip in range(n_period):
            sa_i, sigma_i = BSSA_2014_sub(M, ip, Rjb, U, SS, NS, RS, region, z1, Vs30, full_std=full_std)
            Sa.append(sa_i)
            sigma.append(sigma_i)
            
        Sa, sigma = np.array(Sa), np.array(sigma)    
        return period, Sa, sigma
    else:
        n_T = len(T)
        Sa = []
        sigma = []
        for i in range(len(T)):
            Ti = T[i]
            if Ti not in period:
                index_lower = np.where(period < Ti)[0][-1]
                index_higher = np.where(period > Ti)[0][0]
                T_lo = period[index_lower]
                T_hi = period[index_higher]
                
                if full_std is False:
                    Sa_lo, sigma_lo = BSSA_2014_sub(M, index_lower, Rjb, U, SS, NS, RS, region, z1, Vs30, full_std=full_std)
                    Sa_hi, sigma_hi = BSSA_2014_sub(M, index_higher, Rjb, U, SS, NS, RS, region, z1, Vs30, full_std=full_std)
                    
                    X = [T_lo, T_hi]
                    Y_Sa = [Sa_lo, Sa_hi]
                    Y_sigma = [sigma_lo, sigma_hi]
                    sa_i = np.interp(Ti, X, Y_Sa)
                    sigma_i = np.interp(Ti, X, Y_sigma)
                
                elif full_std is True:
                    Sa_lo, sigma_lo = BSSA_2014_sub(M, index_lower, Rjb, U, SS, NS, RS, region, z1, Vs30, full_std=full_std)
                    Sa_hi, sigma_hi = BSSA_2014_sub(M, index_higher, Rjb, U, SS, NS, RS, region, z1, Vs30, full_std=full_std)
                    
                    X = [T_lo, T_hi]
                    Y_Sa = [Sa_lo, Sa_hi]
                    Y_phi = [sigma_lo[0], sigma_hi[0]]
                    Y_tau = [sigma_lo[1], sigma_hi[1]]
                    
                    sa_i = np.interp(Ti, X, Y_Sa)
                    phi = np.interp(Ti, X, Y_phi)
                    tau = np.interp(Ti, X, Y_tau)
                    sigma_i = [phi, tau]
                
                Sa.append(sa_i)
                sigma.append(sigma_i)
                    
            else:
                index = np.where(period == Ti)[0][0]
                sa_i, sigma_i = BSSA_2014_sub(M, index, Rjb, U, SS, NS, RS, region, z1, Vs30, full_std=full_std)
                Sa.append(sa_i)
                sigma.append(sigma_i)
        
        Sa, sigma = np.array(Sa), np.array(sigma) 
        return Sa, sigma

def BSSA_2014_sub (M, ip, Rjb, U, SS, NS, RS, region, z1, Vs30, full_std=False):
    ## Coefficients
    mref = 4.5
    rref = 1.
    v_ref = 760. 
    f1 = 0.
    f3 = 0.1
    v1 = 225.
    v2 = 300.
    
    period = [0., 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1., 1.5, 2., 3., 4., 5., 7.5, 10.]
    mh = [5.5,5.5,5.5,5.5,5.5,5.5,5.54,5.74,5.92,6.05,6.14,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2]
    e0 = [0.4473,0.4534,0.48598,0.56916,0.75436,0.96447,1.1268,1.3095,1.3255,1.2766,1.2217,1.1046,0.96991,0.66903,0.3932,-0.14954,-0.58669,-1.1898,-1.6388,-1.966,-2.5865,-3.0702]
    e1 = [0.4856,0.4916,0.52359,0.6092,0.79905,1.0077,1.1669,1.3481,1.359,1.3017,1.2401,1.1214,0.99106,0.69737,0.4218,-0.11866,-0.55003,-1.142,-1.5748,-1.8882,-2.4874,-2.9537]
    e2 = [0.2459,0.2519,0.29707,0.40391,0.60652,0.77678,0.8871,1.0648,1.122,1.0828,1.0246,0.89765,0.7615,0.47523,0.207,-0.3138,-0.71466,-1.23,-1.6673,-2.0245,-2.8176,-3.3776]
    e3 = [0.4539,0.4599,0.48875,0.55783,0.72726,0.9563,1.1454,1.3324,1.3414,1.3052,1.2653,1.1552,1.012,0.69173,0.4124,-0.1437,-0.60658,-1.2664,-1.7516,-2.0928,-2.6854,-3.1726]
    e4 = [1.431,1.421,1.4331,1.4261,1.3974,1.4174,1.4293,1.2844,1.1349,1.0166,0.95676,0.96766,1.0384,1.2871,1.5004,1.7622,1.9152,2.1323,2.204,2.2299,2.1187,1.8837]
    e5 = [0.05053,0.04932,0.053388,0.061444,0.067357,0.073549,0.055231,-0.042065,-0.11096,-0.16213,-0.1959,-0.22608,-0.23522,-0.21591,-0.18983,-0.1467,-0.11237,-0.04332,-0.014642,-0.014855,-0.081606,-0.15096 ]
    e6 = [-0.1662,-0.1659,-0.16561,-0.1669,-0.18082,-0.19665,-0.19838,-0.18234,-0.15852,-0.12784,-0.092855,-0.023189,0.029119,0.10829,0.17895,0.33896,0.44788,0.62694,0.76303,0.87314,1.0121,1.0651]
    c1 = [-1.13400,-1.13400,-1.13940,-1.14210,-1.11590,-1.08310,-1.06520,-1.05320,-1.06070,-1.07730,-1.09480,-1.12430,-1.14590,-1.17770,-1.19300,-1.20630,-1.21590,-1.21790,-1.21620,-1.21890,-1.25430,-1.32530]
    c2 = [0.19170,0.19160,0.18962,0.18842,0.18709,0.18225,0.17203,0.15401,0.14489,0.13925,0.13388,0.12512,0.12015,0.11054,0.10248,0.09645,0.09636,0.09764,0.10218,0.10353,0.12507,0.15183]
    c3 = [-0.00809,-0.00809,-0.00807,-0.00834,-0.00982,-0.01058,-0.01020,-0.00898,-0.00772,-0.00652,-0.00548,-0.00405,-0.00322,-0.00193,-0.00121,-0.00037,0.00000,0.00000,-0.00005,0.00000,0.00000,0.00000]
    h = [4.5,4.5,4.5,4.49,4.2,4.04,4.13,4.39,4.61,4.78,4.93,5.16,5.34,5.6,5.74,6.18,6.54,6.93,7.32,7.78,9.48,9.66]
    
    deltac3_gloCATW = [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000]
    deltac3_CHTU = [0.002860,0.002820,0.002780,0.002760,0.002960,0.002960,0.002880,0.002790,0.002610,0.002440,0.002200,0.002110,0.002350,0.002690,0.002920,0.003040,0.002920,0.002620,0.002610,0.002600,0.002600,0.003030]
    deltac3_ITJA = [-0.002550,-0.002440,-0.002340,-0.002170,-0.001990,-0.002160,-0.002440,-0.002710,-0.002970,-0.003140,-0.003300,-0.003210,-0.002910,-0.002530,-0.002090,-0.001520,-0.001170,-0.001190,-0.001080,-0.000570,0.000380,0.001490]
    
    c = [-0.6000,-0.6037,-0.5739,-0.5341,-0.4580,-0.4441,-0.4872,-0.5796,-0.6876,-0.7718,-0.8417,-0.9109,-0.9693,-1.0154,-1.0500,-1.0454,-1.0392,-1.0112,-0.9694,-0.9195,-0.7766,-0.6558]
    vc = [1500.00,1500.20,1500.36,1502.95,1501.42,1494.00,1479.12,1442.85,1392.61,1356.21,1308.47,1252.66,1203.91,1147.59,1109.95,1072.39,1009.49,922.43,844.48,793.13,771.01,775.00]
    f4 = [-0.1500,-0.1483,-0.1471,-0.1549,-0.1963,-0.2287,-0.2492,-0.2571,-0.2466,-0.2357,-0.2191,-0.1958,-0.1704,-0.1387,-0.1052,-0.0679,-0.0361,-0.0136,-0.0032,-0.0003,-0.0001,0.0000]
    f5 = [-0.00701,-0.00701,-0.00728,-0.00735,-0.00647,-0.00573,-0.00560,-0.00585,-0.00614,-0.00644,-0.00670,-0.00713,-0.00744,-0.00812,-0.00844,-0.00771,-0.00479,-0.00183,-0.00152,-0.00144,-0.00137,-0.00136]
    f6 = [-9.900,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,0.092,0.367,0.638,0.871,1.135,1.271,1.329,1.329,1.183]
    f7 = [-9.900,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,-9.9,0.059,0.208,0.309,0.382,0.516,0.629,0.738,0.809,0.703]
    tau1 = [0.3980,0.4020,0.4090,0.4450,0.5030,0.4740,0.4150,0.3540,0.3440,0.3500,0.3630,0.3810,0.4100,0.4570,0.4980,0.5250,0.5320,0.5370,0.5430,0.5320,0.5110,0.4870]
    tau2 = [0.3480,0.3450,0.3460,0.3640,0.4260,0.4660,0.4580,0.3880,0.3090,0.2660,0.2290,0.2100,0.2240,0.2660,0.2980,0.3150,0.3290,0.3440,0.3490,0.3350,0.2700,0.2390]
    phi1 = [0.6950,0.6980,0.7020,0.7210,0.7530,0.7450,0.7280,0.7200,0.7110,0.6980,0.6750,0.6430,0.6150,0.5810,0.5530,0.5320,0.5260,0.5340,0.5360,0.5280,0.5120,0.5100]
    phi2 = [0.4950,0.4990,0.5020,0.5140,0.5320,0.5420,0.5410,0.5370,0.5390,0.5470,0.5610,0.5800,0.5990,0.6220,0.6250,0.6190,0.6180,0.6190,0.6160,0.6220,0.6340,0.6040]
    dphiR = [0.100,0.096,0.092,0.081,0.063,0.064,0.087,0.120,0.136,0.141,0.138,0.122,0.109,0.100,0.098,0.104,0.105,0.088,0.070,0.061,0.058,0.060]
    dphiV = [0.070,0.070,0.030,0.029,0.030,0.022,0.014,0.015,0.045,0.055,0.050,0.049,0.060,0.070,0.020,0.010,0.008,0.000,0.000,0.000,0.000,0.000]
    R1 = [110.000,111.670,113.100,112.130,97.930,85.990,79.590,81.330,90.910,97.040,103.150,106.020,105.540,108.390,116.390,125.380,130.370,130.360,129.490,130.220,130.720,130.000]
    R2 = [270.000,270.000,270.000,270.000,270.000,270.040,270.090,270.160,270.000,269.450,268.590,266.540,265.000,266.510,270.000,262.410,240.140,195.000,199.450,230.000,250.390,210.000]
    
    ## The source(event function):
    if M <= mh [ip]:
        F_E = e0[ip] * U + e1[ip] * SS + e2[ip] * NS + e3[ip] * RS + e4[ip] * (M - mh[ip]) + e5[ip] * (M - mh[ip])**2.
    else:
        F_E = e0[ip] * U + e1[ip] * SS + e2[ip] * NS + e3[ip] * RS + e6[ip] * (M - mh[ip])
    
    ## The path function:
    if region == 0 or region == 1:
        deltac3 = deltac3_gloCATW
    elif region == 3:
        deltac3 = deltac3_CHTU
    elif region == 2 or region == 4:
        deltac3 = deltac3_ITJA
    
    r = np.sqrt(Rjb**2. + h[ip]**2.)
    F_P= (c1[ip] + c2[ip] * (M - mref)) * np.log (r / rref) + (c3[ip] + deltac3[ip]) * (r - rref)
    
    ## FIND PGAr
    if Vs30 != v_ref or ip != 0:
        PGA_r, sigma_r = BSSA_2014_sub(M, 0, Rjb, U, SS, NS, RS, region, z1, v_ref)
    
    ## The site function:
    # Linear component
        if Vs30<= vc[ip]:
            ln_Flin = c[ip] * np.log(Vs30 / v_ref)
        else:
            ln_Flin = c[ip] * np.log(vc[ip]/ v_ref)
        
    # Nonlinear component
        f2 = f4[ip]*(np.exp(f5[ip]*(min(Vs30, 760.)-360.))-np.exp(f5[ip]*(760. - 360.)))
        ln_Fnlin = f1 + f2*np.log((PGA_r + f3)/f3)
        
    # Effect of basin depth
        if z1 != 999:
            if region == 1:  # if in California
                mu_z1 = np.exp(-7.15/4.*np.log((Vs30**4. + 570.94**4.)/(1360.**4. + 570.94**4.)))/1000.
            elif region == 2:  # if in Japan
                    mu_z1 = np.exp(-5.23/2. * np.log((Vs30**2. + 412.39**2.)/(1360**2.+412.39**2.)))/1000.
            else:
                mu_z1 = np.exp(-7.15/4. * np.log((Vs30**4. + 570.94**4.)/(1360.**4. + 570.94**4.)))/1000.

            dz1 = z1 - mu_z1
        else:
            dz1 = 0.

        
        if z1 != 999:
            if period[ip] < 0.65:
                F_dz1 = 0
            elif period[ip] >= 0.65 and abs(dz1) <= f7[ip]/f6[ip]:
                F_dz1 = f6[ip]*dz1
            else:
                F_dz1 = f7[ip]
        else:
            F_dz1 = 0
        
        F_S = ln_Flin + ln_Fnlin + F_dz1
        ln_Y = F_E + F_P + F_S
        median = np.exp(ln_Y)
    
    else:
        ln_y = F_E + F_P
        median = np.exp(ln_y)
    
    ## Aleatory - uncertainty function
    if M <= 4.5:
        tau = tau1[ip]
        phi_M = phi1[ip]
    elif 4.5 < M and M < 5.5:
        tau = tau1[ip] + (tau2[ip]-tau1[ip])*(M-4.5)
        phi_M = phi1[ip] + (phi2[ip]-phi1[ip])*(M-4.5)
    elif M >= 5.5:
        tau = tau2[ip]
        phi_M = phi2[ip]

    if Rjb <= R1[ip]:
        phi_MR = phi_M
    elif (R1[ip] < Rjb and Rjb <= R2[ip]):
        phi_MR = phi_M + dphiR[ip]*(np.log(Rjb/R1[ip])/np.log(R2[ip]/R1[ip]))
    elif Rjb > R2[ip]:
        phi_MR = phi_M + dphiR[ip]
            
    if Vs30 >= v2:
        phi_MRV = phi_MR
    elif (v1 <= Vs30 and Vs30 <= v2):
        phi_MRV = phi_MR - dphiV[ip]*(np.log(v2/Vs30)/np.log(v2/v1))
    elif Vs30 <= v1:
        phi_MRV = phi_MR - dphiV[ip]

    if full_std is False:
        sigma = np.sqrt(phi_MRV**2. + tau**2.)
    elif full_std is True:
        sigma = [phi_MRV, tau]
    
    return median, sigma
########################################################################################################################
def CB_2014(M, T, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, gamma, Fhw, Vs30, Z25, Zhyp, region, full_std=False):
    # Campbell and Bozorgnia 2014 ground motion prediciton model. Citation for
    # the model:
    #
    # Campbell, K. W., and Bozorgnia, Y. (2014). "NGA-West2 Ground Motion Model 
    # for the Average Horizontal Components of PGA, PGV, and 5# Damped Linear 
    # Acceleration Response Spectra." Earthquake Spectra, 30(3), 1087-1115.
    #
    # coded by Yue Hua,  4/22/14
    #       Stanford University, yuehua@stanford.edu
    # Modified 4/5/2015 by Jack Baker to fix a few small bugs
    # Modified 12/9/2019 by Jack Baker to update a few c6 coefficients that changed from the PEER report to the Earthquake Spectra paper (thank you Yenan Cao for noting this)
    #
    # Provides ground-mtion prediction equations for computing medians and
    # standard deviations of average horizontal components of PGA, PGV and 5#
    # damped linear pseudo-absolute aceeleration response spectra
    #
    ###########################################################################
    # Input Variables
    # M             = Magnitude
    # T             = Period (sec);
    #                 Use 1000 for output the array of Sa with period
    # Rrup          = Closest distance coseismic rupture (km)
    # Rjb           = Joyner-Boore distance (km)
    # Rx            = Closest distance to the surface projection of the
    #                   coseismic fault rupture plane
    # W             = down-dip width of the fault rupture plane
    #                   if unknown, input: 999
    # Ztor          = Depth to the top of coseismic rupture (km)
    #                   if unknown, input: 999
    # Zbot          = Depth to the bottom of the seismogenic crust
    #               needed only when W is unknow;
    # delta         = average dip of the rupture place (degree)
    # gamma        = rake angle (degree) - average angle of slip measured in
    #                   the plance of rupture
    # Fhw           = hanging wall effect
    #               = 1 for including
    #               = 0 for excluding
    # Vs30          = shear wave velocity averaged over top 30 m (m/s)
    # Z25           = Depth to the 2.5 km/s shear-wave velocity horizon (km)
    #                   if in California or Japan and Z2.5 is unknow, then
    #                   input: 999
    # Zhyp          = Hypocentral depth of the earthquake measured from sea level
    #                   if unknown, input: 999
    # region        = 0 for global (incl. Taiwan)
    #               = 1 for California
    #               = 2 for Japan
    #               = 3 for China or Turkey
    #               = 4 for Italy
    # Output Variables
    # Sa            = Median spectral acceleration prediction
    # sigma         = logarithmic standard deviation of spectral acceleration
    #                 prediction
    ###########################################################################
    # Period
    period = np.array([0.010, 0.020, 0.030, 0.050, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 0, -1])
    n_period = len(period)
    
    # Set initial A1100 value to 999. A1100: median estimated value of PGA on rock with Vs30 = 1100m/s
    A1100 = 999
    
    # Style of faulting
    if gamma > 30 and gamma < 150:
        Frv = 1 
    else: 
        Frv = 0
        
    if gamma > -150 and gamma < -30:
        Fnm = 1 
    else: 
        Fnm = 0

    # if Ztor is unknown..
    if Ztor == 999:
        if Frv == 1:
            Ztor = max(2.704 - 1.226*max(M-5.849,0),0)**2.
        else:
            Ztor = max(2.673 - 1.136*max(M-4.970,0),0)**2.
    
    # if W is unknown...
    if W == 999:
        if Frv == 1:
            Ztori = max(2.704 - 1.226*max(M-5.849,0),0)**2
        else:
            Ztori = max(2.673 - 1.136*max(M-4.970,0),0)**2
            W = min(np.sqrt(10**((M - 4.07)/0.98)),(Zbot - Ztori)/np.sin(np.pi/180.*delta))
            Zhyp = 9
    
    # if Zhyp is unknown...
    if Zhyp == 999 and W != 999:
        if M < 6.75:
            fdZM = -4.317 + 0.984*M
        else:
            fdZM = 2.325
        
        if delta <= 40:
            fdZD = 0.0445*(delta-40)
        else:
            fdZD = 0
    
        if Frv == 1:
            Ztori = max(2.704 - 1.226*max(M-5.849,0),0)**2.
        else:
            Ztori = max(2.673 - 1.136*max(M-4.970,0),0)**2.
        
        Zbor = Ztori + W*np.sin(np.pi/180.*delta)
        d_Z = np.exp(min(fdZM + fdZD,np.log(0.9*(Zbor-Ztori))))
        Zhyp = d_Z + Ztori


    # Compute Sa and sigma with pre-defined period
    if type(T) is int and T == 1000:
        Sa = np.zeros(n_period-2)
        sigma = np.zeros(n_period-2)
        period1 = period[0:-2]
        for ipT in range(len(period1)):
            Sa[ipT], sigma[ipT] = CB_2014_nga_sub(M, ipT, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
            PGA, ee = CB_2014_nga_sub(M, 21, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
            if Sa[ipT] < PGA and period1[ipT] < 0.25:
                Sa[ipT] = PGA
            
    # Compute Sa and sigma with user-defined period    
    else:                 
        Sa = np.zeros(len(T))
        sigma = np.zeros(len(T))
        period1 = T
        for i in range(len(T)):
            Ti = T[i]
            if Ti not in period: # The user defined period requires interpolation
                ip_low = np.where(period < Ti)[0][-1]
                ip_high = np.where(period > Ti)[0][0]
                T_low = period[ip_low]
                T_high = period[ip_high]
                
                
                if full_std is False:
                    Sa_low, sigma_low = CB_2014_nga_sub(M, ip_low, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
                    Sa_high, sigma_high = CB_2014_nga_sub(M, ip_high, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
                    PGA, ee = CB_2014_nga_sub(M, 21, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
                
                    x = [np.log(T_low), np.log(T_high)]
                    Y_sa = [np.log(Sa_low), np.log(Sa_high)]
                    Y_sigma = [sigma_low, sigma_high]
                    Sa[i] = np.exp(np.interp(np.log(Ti), x, Y_sa))
                    sigma[i] = np.interp(np.log(Ti), x, Y_sigma)
                    
                elif full_std is True:
                    Sa_low, sigma_low = CB_2014_nga_sub(M, ip_low, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
                    Sa_high, sigma_high = CB_2014_nga_sub(M, ip_high, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
                    PGA, ee = CB_2014_nga_sub(M, 21, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
                
                    x = [np.log(T_low), np.log(T_high)]
                    Y_sa = [np.log(Sa_low), np.log(Sa_high)]
                    Y_phi = [sigma_low[0], sigma_high[0]]
                    Y_tau = [sigma_low[1], sigma_high[1]]
                    Sa[i] = np.exp(np.interp(np.log(Ti), x, Y_sa))
                    phi = np.interp(np.log(Ti), x, Y_phi)
                    tau = np.interp(np.log(Ti), x, Y_tau)
                    sigma[i] = [phi, tau]
                
                if Sa[i] < PGA and period1[i] < 0.25:
                    Sa[i] = PGA
            else:
                ip_T = np.where(period == Ti)[0][0]
                Sa[i], sigma[i] = CB_2014_nga_sub(M, ip_T, Rrup, Rjb, Rx, W, Ztor,Zbot,delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
                PGA, ee = CB_2014_nga_sub(M, 21, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25, Zhyp,gamma, Frv, Fnm, region, A1100, full_std=full_std)
                
                if Sa[i] < PGA and period1[i] < 0.25:
                    Sa[i] = PGA
                    
    return period1, Sa, sigma
 
def CB_2014_nga_sub(M, ip, Rrup, Rjb, Rx, W, Ztor, Zbot, delta, Fhw, Vs30, Z25in, Zhyp, gamma,Frv, Fnm, region, A1100, full_std=False):

    c0 = [-4.365, -4.348, -4.024, -3.479, -3.293, -3.666, -4.866, -5.411, -5.962, -6.403, -7.566, -8.379, -9.841, -11.011, -12.469, -12.969, -13.306, -14.02, -14.558, -15.509, -15.975, -4.416, -2.895]
    c1 = [0.977, 0.976, 0.931, 0.887, 0.902, 0.993, 1.267, 1.366, 1.458, 1.528, 1.739, 1.872, 2.021, 2.180, 2.270, 2.271, 2.150, 2.132, 2.116, 2.223, 2.132, 0.984, 1.510]
    c2 = [0.533, 0.549, 0.628, 0.674, 0.726, 0.698, 0.510, 0.447, 0.274, 0.193, -0.020, -0.121, -0.042, -0.069, 0.047, 0.149, 0.368, 0.726, 1.027, 0.169, 0.367, 0.537, 0.270]
    c3 = [-1.485, -1.488, -1.494, -1.388, -1.469, -1.572, -1.669, -1.750, -1.711, -1.770, -1.594, -1.577, -1.757, -1.707, -1.621, -1.512, -1.315, -1.506, -1.721, -0.756, -0.800, -1.499, -1.299]
    c4 = [-0.499, -0.501, -0.517, -0.615, -0.596, -0.536, -0.490, -0.451, -0.404, -0.321, -0.426, -0.440, -0.443, -0.527, -0.630, -0.768, -0.890, -0.885, -0.878, -1.077, -1.282, -0.496, -0.453]
    c5 = [-2.773, -2.772, -2.782, -2.791, -2.745, -2.633, -2.458, -2.421, -2.392, -2.376, -2.303, -2.296, -2.232, -2.158, -2.063, -2.104, -2.051, -1.986, -2.021, -2.179, -2.244, -2.773, -2.466]
    #c6 = [0.248, 0.247, 0.246, 0.240, 0.227, 0.210, 0.183, 0.182, 0.189, 0.195, 0.185, 0.186, 0.186, 0.169, 0.158, 0.158, 0.148, 0.135, 0.140, 0.178, 0.194, 0.248, 0.204]; # these coefficients are in the PEER report
    c6 = [0.248, 0.247, 0.246, 0.240, 0.227, 0.210, 0.183, 0.182, 0.189, 0.195, 0.185, 0.186, 0.186, 0.169, 0.158, 0.158, 0.148, 0.135, 0.135, 0.165, 0.180, 0.248, 0.204]# these coefficients are in the Earthquake Spectra paper
    c7 = [6.753, 6.502, 6.291, 6.317, 6.861, 7.294, 8.031, 8.385, 7.534, 6.990, 7.012, 6.902, 5.522, 5.650, 5.795, 6.632, 6.759, 7.978, 8.538, 8.468, 6.564, 6.768, 5.837]
    c8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c9 = [-0.214, -0.208, -0.213, -0.244, -0.266, -0.229, -0.211, -0.163, -0.150, -0.131, -0.159, -0.153, -0.090, -0.105, -0.058, -0.028, 0, 0, 0, 0, 0, -0.212, -0.168]
    c10 = [0.720, 0.730, 0.759, 0.826, 0.815, 0.831, 0.749, 0.764, 0.716, 0.737, 0.738, 0.718, 0.795, 0.556, 0.480, 0.401, 0.206, 0.105, 0, 0, 0, 0.720, 0.305]
    c11 = [1.094, 1.149, 1.290, 1.449, 1.535, 1.615, 1.877, 2.069, 2.205, 2.306, 2.398, 2.355, 1.995, 1.447, 0.330, -0.514, -0.848, -0.793, -0.748, -0.664, -0.576, 1.090, 1.713]
    c12 = [2.191, 2.189, 2.164, 2.138, 2.446, 2.969, 3.544, 3.707, 3.343, 3.334, 3.544, 3.016, 2.616, 2.470, 2.108, 1.327, 0.601, 0.568, 0.356, 0.075, -0.027, 2.186, 2.602]
    c13 = [1.416, 1.453, 1.476, 1.549, 1.772, 1.916, 2.161, 2.465, 2.766, 3.011, 3.203, 3.333, 3.054, 2.562, 1.453, 0.657, 0.367, 0.306, 0.268, 0.374, 0.297, 1.420, 2.457]
    c14 = [-0.0070, -0.0167, -0.0422, -0.0663, -0.0794, -0.0294, 0.0642, 0.0968, 0.1441, 0.1597, 0.1410, 0.1474, 0.1764, 0.2593, 0.2881, 0.3112, 0.3478, 0.3747, 0.3382, 0.3754, 0.3506, -0.0064, 0.1060]
    c15 = [-0.207, -0.199, -0.202, -0.339, -0.404, -0.416, -0.407, -0.311, -0.172, -0.084, 0.085, 0.233, 0.411, 0.479, 0.566, 0.562, 0.534, 0.522, 0.477, 0.321, 0.174, -0.202, 0.332]
    c16 = [0.390, 0.387, 0.378, 0.295, 0.322, 0.384, 0.417, 0.404, 0.466, 0.528, 0.540, 0.638, 0.776, 0.771, 0.748, 0.763, 0.686, 0.691, 0.670, 0.757, 0.621, 0.393, 0.585]
    c17 = [0.0981, 0.1009, 0.1095, 0.1226, 0.1165, 0.0998, 0.0760, 0.0571, 0.0437, 0.0323, 0.0209, 0.0092, -0.0082, -0.0131, -0.0187, -0.0258, -0.0311, -0.0413, -0.0281, -0.0205, 0.0009, 0.0977, 0.0517]
    c18 = [0.0334, 0.0327, 0.0331, 0.0270, 0.0288, 0.0325, 0.0388, 0.0437, 0.0463, 0.0508, 0.0432, 0.0405, 0.0420, 0.0426, 0.0380, 0.0252, 0.0236, 0.0102, 0.0034, 0.0050, 0.0099, 0.0333, 0.0327]
    c19 = [0.00755, 0.00759, 0.00790, 0.00803, 0.00811, 0.00744, 0.00716, 0.00688, 0.00556, 0.00458, 0.00401, 0.00388, 0.00420, 0.00409, 0.00424, 0.00448, 0.00345, 0.00603, 0.00805, 0.00280, 0.00458, 0.00757, 0.00613]
    c20 = [-0.0055, -0.0055, -0.0057, -0.0063, -0.0070, -0.0073, -0.0069, -0.0060, -0.0055, -0.0049, -0.0037, -0.0027, -0.0016, -0.0006, 0, 0, 0, 0, 0, 0, 0, -0.0055, -0.0017]
    Dc20 =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Dc20_JI = [-0.0035, -0.0035, -0.0034, -0.0037, -0.0037, -0.0034, -0.0030, -0.0031, -0.0033, -0.0035, -0.0034, -0.0034, -0.0032, -0.0030, -0.0019, -0.0005, 0, 0, 0, 0, 0, -0.0035, -0.0006]
    Dc20_CH = [0.0036, 0.0036, 0.0037, 0.0040, 0.0039, 0.0042, 0.0042, 0.0041, 0.0036, 0.0031, 0.0028, 0.0025, 0.0016, 0.0006, 0, 0, 0, 0, 0, 0, 0, 0.0036, 0.0017]
    a2 = [0.168, 0.166, 0.167, 0.173, 0.198, 0.174, 0.198, 0.204, 0.185, 0.164, 0.160, 0.184, 0.216, 0.596, 0.596, 0.596, 0.596, 0.596, 0.596, 0.596, 0.596, 0.167, 0.596]
    h1 = [0.242, 0.244, 0.246, 0.251, 0.260, 0.259, 0.254, 0.237, 0.206, 0.210, 0.226, 0.217, 0.154, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.241, 0.117]
    h2 = [1.471, 1.467, 1.467, 1.449, 1.435, 1.449, 1.461, 1.484, 1.581, 1.586, 1.544, 1.554, 1.626, 1.616, 1.616, 1.616, 1.616, 1.616, 1.616, 1.616, 1.616, 1.474, 1.616]
    h3 = [-0.714, -0.711, -0.713, -0.701, -0.695, -0.708, -0.715, -0.721, -0.787, -0.795, -0.770, -0.770, -0.780, -0.733, -0.733, -0.733, -0.733, -0.733, -0.733, -0.733, -0.733, -0.715, -0.733]
    h4 = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
    h5 = [-0.336, -0.339, -0.338, -0.338, -0.347, -0.391, -0.449, -0.393, -0.339, -0.447, -0.525, -0.407, -0.371, -0.128, -0.128, -0.128, -0.128, -0.128, -0.128, -0.128, -0.128, -0.337, -0.128]
    h6 = [-0.270, -0.263, -0.259, -0.263, -0.219, -0.201, -0.099, -0.198, -0.210, -0.121, -0.086, -0.281, -0.285, -0.756, -0.756, -0.756, -0.756, -0.756, -0.756, -0.756, -0.756, -0.270, -0.756]
    k1 = [865, 865, 908, 1054, 1086, 1032, 878, 748, 654, 587, 503, 457, 410, 400, 400, 400, 400, 400, 400, 400, 400, 865, 400]
    k2 = [-1.186, -1.219, -1.273, -1.346, -1.471, -1.624, -1.931, -2.188, -2.381, -2.518, -2.657, -2.669, -2.401, -1.955, -1.025, -0.299, 0.000, 0.000, 0.000, 0.000, 0.000, -1.186, -1.955]
    k3 = [1.839, 1.840, 1.841, 1.843, 1.845, 1.847, 1.852, 1.856, 1.861, 1.865, 1.874, 1.883, 1.906, 1.929, 1.974, 2.019, 2.110, 2.200, 2.291, 2.517, 2.744, 1.839, 1.929]
    c = 1.88
    n = 1.18
    f1 = [0.734, 0.738, 0.747, 0.777, 0.782, 0.769, 0.769, 0.761, 0.744, 0.727, 0.690, 0.663, 0.606, 0.579, 0.541, 0.529, 0.527, 0.521, 0.502, 0.457, 0.441, 0.734, 0.655]
    f2 = [0.492, 0.496, 0.503, 0.520, 0.535, 0.543, 0.543, 0.552, 0.545, 0.568, 0.593, 0.611, 0.633, 0.628, 0.603, 0.588, 0.578, 0.559, 0.551, 0.546, 0.543, 0.492, 0.494]
    t1 = [0.404, 0.417, 0.446, 0.508, 0.504, 0.445, 0.382, 0.339, 0.340, 0.340, 0.356, 0.379, 0.430, 0.470, 0.497, 0.499, 0.500, 0.543, 0.534, 0.523, 0.466, 0.409, 0.317]
    t2 = [0.325, 0.326, 0.344, 0.377, 0.418, 0.426, 0.387, 0.338, 0.316, 0.300, 0.264, 0.263, 0.326, 0.353, 0.399, 0.400, 0.417, 0.393, 0.421, 0.438, 0.438, 0.322, 0.297]
    flnAF = [0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300]
    rlnPGA_lnY = [1.000, 0.998, 0.986, 0.938, 0.887, 0.870, 0.876,0.870, 0.850, 0.819, 0.743, 0.684, 0.562, 0.467, 0.364, 0.298, 0.234, 0.202, 0.184, 0.176, 0.154, 1.000, 0.684]
    
    # Adjustment factor based on region
    if region == 2:
        Dc20 = Dc20_JI
    elif region == 4:
        Dc20 = Dc20_JI
    elif region == 3:
        Dc20 = Dc20_CH

    # if region is in Japan...
    if region == 2:
        Sj = 1.
    else:
        Sj = 0.
    
    # if Z2.5 is unknown...
    if Z25in == 999:
        if region != 2:  # if in California or other locations
            Z25 = np.exp(7.089 - 1.144 * np.log(Vs30))
            Z25A = np.exp(7.089 - 1.144 * np.log(1100))
        elif region == 2:  # if in Japan
            Z25 = np.exp(5.359 - 1.102 * np.log(Vs30))
            Z25A = np.exp(5.359 - 1.102 * np.log(1100))
    else:
    # Assign Z2.5 from user input into Z25 and calc Z25A for Vs30=1100m/s
        if region != 2:  # if in California or other locations
            Z25 = Z25in
            Z25A = np.exp(7.089 - 1.144 * np.log(1100))
        elif region == 2:  # if in Japan
            Z25 = Z25in
            Z25A = np.exp(5.359 - 1.102 * np.log(1100))

    # Magnitude dependence
    if M <= 4.5:
        fmag = c0[ip] + c1[ip] * M
    elif M <= 5.5:
        fmag = c0[ip] + c1[ip] * M + c2[ip] * (M - 4.5)
    elif M <= 6.5:
        fmag = c0[ip] + c1[ip] * M + c2[ip] * (M - 4.5) + c3[ip] * (M - 5.5)
    else:
        fmag = c0[ip] + c1[ip] * M + c2[ip] * (M - 4.5) + c3[ip] * (M - 5.5) + c4[ip] * (M-6.5)

    ## Geometric attenuation term
    fdis = (c5[ip] + c6[ip] * M) * np.log(np.sqrt(Rrup**2. + c7[ip]**2.))
    
    ## Style of faulting
    if M <= 4.5:
        F_fltm = 0
    elif M<= 5.5:
        F_fltm = M-4.5
    else:
        F_fltm = 1
    
    fflt = ((c8[ip] * Frv) + (c9[ip] * Fnm))*F_fltm
    ## Hanging-wall effects
    R1 = W * np.cos(np.pi/180.*delta) # W - downdip width
    R2 = 62.* M - 350.
    
    f1_Rx = h1[ip]+h2[ip]*(Rx/R1)+h3[ip]*(Rx/R1)**2.
    f2_Rx = h4[ip]+h5[ip]*((Rx-R1)/(R2-R1))+h6[ip]*((Rx-R1)/(R2-R1))**2.
    
    if Fhw == 0:
        f_hngRx = 0
    elif Rx < R1 and Fhw == 1:
        f_hngRx = f1_Rx
    elif Rx >= R1 and Fhw == 1:
        f_hngRx = max(f2_Rx,0)
    
    if Rrup == 0:
        f_hngRup = 1
    else:
        f_hngRup = (Rrup-Rjb)/Rrup
    
    if M <= 5.5:
        f_hngM = 0
    elif M <= 6.5:
        f_hngM = (M-5.5)*(1+a2[ip]*(M-6.5))
    else:
        f_hngM = 1+a2[ip]*(M-6.5)
    
    if Ztor <= 16.66:
        f_hngZ = 1-0.06*Ztor
    else:
        f_hngZ = 0
    
    f_hngdelta = (90. - delta)/45.
    fhng = c10[ip] * f_hngRx * f_hngRup * f_hngM * f_hngZ * f_hngdelta

    ## Site conditions
    if Vs30 <= k1[ip]:
        if A1100 == 999:
            A1100 = CB_2014(M, [0], Rrup, Rjb, Rx, W, Ztor, Zbot, delta, gamma, Fhw, 1100, Z25A, Zhyp, region)
        f_siteG = c11[ip] * np.log(Vs30/k1[ip]) + k2[ip] * (np.log(A1100[1] + c * (Vs30/k1[ip])**n) - np.log(A1100[1] + c))
        
    elif Vs30 > k1[ip]:
        f_siteG = (c11[ip] + k2[ip] * n) * np.log(Vs30/k1[ip])
    
    if Vs30 <= 200:
        f_siteJ = (c12[ip]+k2[ip]*n)*(np.log(Vs30/k1[ip])-np.log(200/k1[ip]))*Sj
    else:
        f_siteJ = (c13[ip]+k2[ip]*n)*np.log(Vs30/k1[ip])*Sj
    
    fsite = f_siteG + f_siteJ
    
    ## Basin Response Term - Sediment effects
    if Z25 <= 1:
        fsed = (c14[ip]+c15[ip]*Sj) * (Z25 - 1)
    elif Z25 <= 3:
        fsed = 0
    elif Z25 > 3:
        fsed = c16[ip] * k3[ip] * np.exp(-0.75) * (1. - np.exp(-0.25 * (Z25 - 3.)))
    
    ## Hypocenteral Depth term
    if Zhyp <= 7:
        f_hypH = 0
    elif Zhyp <= 20.:
        f_hypH = Zhyp - 7.
    else:
        f_hypH = 13
    
    if M <= 5.5:
        f_hypM = c17[ip]
    elif M <= 6.5:
        f_hypM = c17[ip]+ (c18[ip]-c17[ip])*(M-5.5)
    else:
        f_hypM = c18[ip]
    
    fhyp = f_hypH * f_hypM
    
    ## Fault Dip term
    if M <= 4.5:
        f_dip = c19[ip]* delta
    elif M <= 5.5:
        f_dip = c19[ip]*(5.5-M)* delta
    else:
        f_dip = 0
    
    ## Anelastic Attenuation Term
    
    if Rrup > 80.:
        f_atn = (c20[ip] + Dc20[ip])*(Rrup-80.)
    else:
        f_atn = 0
    
    ## Median value
    Sa = np.exp(fmag + fdis + fflt + fhng + fsite + fsed + fhyp + f_dip + f_atn)
    
    ## Standard deviation computations
    
    if M <= 4.5:
        tau_lny = t1[ip]
        tau_lnPGA = t1[21]   # ip = PGA
        phi_lny = f1[ip]
        phi_lnPGA = f1[21]
    elif M < 5.5:
        tau_lny = t2[ip]+(t1[ip]-t2[ip])*(5.5-M)
        tau_lnPGA = t2[21]+(t1[21]-t2[21])*(5.5-M)    #ip = PGA
        phi_lny = f2[ip]+(f1[ip]-f2[ip])*(5.5-M)
        phi_lnPGA = f2[21]+(f1[21]-f2[21])*(5.5-M)
    else:
        tau_lny = t2[ip]
        tau_lnPGA = t2[21]
        phi_lny = f2[ip]
        phi_lnPGA = f2[21]
    
    tau_lnyB = tau_lny
    tau_lnPGAB = tau_lnPGA
    phi_lnyB = np.sqrt(phi_lny**2. - flnAF[ip]**2.)
    phi_lnPGAB = np.sqrt(phi_lnPGA**2.-flnAF[ip]**2.)
    
    if Vs30 < k1[ip]:
        alpha = k2[ip] * A1100[1] * ((A1100[1] + c*(Vs30/k1[ip])**n)**(-1) - (A1100[1] + c)**(-1))
    else:
        alpha = 0
    
    tau = np.sqrt(tau_lnyB**2. + alpha**2.*tau_lnPGAB**2. + 2.*alpha*rlnPGA_lnY[ip]*tau_lnyB*tau_lnPGAB)
    phi = np.sqrt(phi_lnyB**2. + flnAF[ip]**2. + alpha**2.*phi_lnPGAB**2. + 2.*alpha*rlnPGA_lnY[ip]*phi_lnyB*phi_lnPGAB)
    if full_std is False:
        sigma = np.sqrt(tau**2. + phi**2.)
    else:
        sigma = [phi, tau]
    return Sa, sigma

# ----------------------------------------------------------------------------------------------------------------------- #
def Bindi2014(M, T, R, vs30, sof, distance='Rjb'):
    # Bind et al 2014 ground motion prediction model for Europe
    # the model:
    #
    # Provides ground-mtion prediction equations for computing medians and
    # standard deviations of average horizontal components of PGA, PGV and 5#
    # damped linear pseudo-absolute aceeleration response spectra
    #
    ###########################################################################
    # Input Variables
    # M             = Magnitude
    # T             = Period (sec);
    #                 Use 1000 for output the array of Sa with period
    # Rhypo         = Hypocentral distance (to a maximum of 300 km)
    # Vs30          = shear wave velocity averaged over top 30 m (m/s)
    # sof           = Style of faulting (S: strike slip, N: Normal, R:reverse)
    # 
    # Output Variables
    # Sa            = Median spectral acceleration prediction (cm/s2)
    # sigma         = logarithmic standard deviation of spectral acceleration
    #                 prediction
    ###########################################################################
    #period = [0.0, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.26, 0.3, 0.36, 0.4, 0.46, 0.5, 0.6, 0.7, 0.9, 1., 1.3, 1.5, 1.8, 2., 2.6, 3.]
    period = np.array([0., 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.26, 0.3, 0.36, 0.4, 0.46, 0.5, 0.7, 0.9, 1., 1.3, 1.5, 1.8, 2., 2.6, 3.])
    n_period = len(period)
    
     # Compute Sa and sigma with pre-defined period
    if type(T) is int and T == 1000:
        Sa = np.zeros(n_period)
        sigma = np.zeros(n_period)
        for ipT in range(n_period):
            Sa[ipT], sigma[ipT] = Bindi2014_sub(M, ipT, R, vs30, sof, distance='Rjb')
        return period, Sa, sigma
    else:
        Sa = np.zeros(len(T))
        sigma = np.zeros(len(T))
        period1 = T
        for i in range(len(T)):
            Ti = T[i]
            if Ti not in period: # The user defined period requires interpolation
                ip_low = np.where(period < Ti)[0][-1]
                ip_high = np.where(period > Ti)[0][0]
                T_low = period[ip_low]
                T_high = period[ip_high]
                
                Sa_low, sigma_low = Bindi2014_sub(M, ip_low, R, vs30, sof, distance='Rjb')
                Sa_high, sigma_high = Bindi2014_sub(M, ip_low, R, vs30, sof, distance='Rjb')
                x = [np.log(T_low), np.log(T_high)]
                Y_sa = [np.log(Sa_low), np.log(Sa_high)]
                Y_sigma = [sigma_low, sigma_high]
                Sa[i] = np.exp(np.interp(np.log(Ti), x, Y_sa))
                sigma[i] = np.interp(np.log(Ti), x, Y_sigma)

            else:
                ipT = np.where(period == Ti)[0][0]
                Sa[i], sigma[i] = Bindi2014_sub(M, ipT, R, vs30, sof, distance='Rjb')
        return Sa, sigma

def Bindi2014_sub(M, ip, R, vs30, sof, distance='Rjb'):
    if distance == 'Rjb':
        e1 = [3.32819, 3.37053, 3.43922, 3.59651, 3.68638, 3.68632, 3.68262, 3.64314, 3.63985, 3.5748, 3.53006, 3.43387, 3.40554, 3.30442, 3.23882, 3.1537, 3.13481, 3.12474,
            2.89841, 2.84727, 2.68016, 2.60171, 2.39067]  
        
        c1 = [-1.2398, -1.26358, -1.31025, -1.29051, -1.28178, -1.17697, -1.10301, -1.08527, -1.10591, -1.09955, -1.09538, -1.06586, -1.05767, -1.05014, -1.05021, -1.04654,
            -1.04612, -1.0527, -0.97383, -0.98339, -0.98308, -0.97922, -0.97753] 

        c2 = [0.21732, 0.220527, 0.244676, 0.231878, 0.219406, 0.182662, 0.133154, 0.115603, 0.108276, 0.103083, 0.101111, 0.109066, 0.112197, 0.121734, 0.114674, 0.129522,
            0.114536, 0.103471, 0.104898, 0.109072, 0.164027, 0.163344, 0.211831]

        h = [5.26486, 5.20082, 4.91669, 5.35922, 6.12146, 5.74154, 5.31998, 5.13455, 5.12846, 4.90557, 4.95386, 4.6599, 4.43205, 4.21657, 4.17127, 4.20016, 4.48003, 4.41613,
            4.25821, 4.56697, 4.68008, 4.58186, 5.39517]
        
        c3 = [0.001186, 0.001118, 0.001092, 0.001821, 0.002114, 0.00254, 0.002421, 0.001964, 0.001499, 0.001049, 0.000851, 0.000868, 0.000789, 0.000487, 0.000159, 0, 0, 0, 0, 0,
            0, 0, 0 ]
        
        b1 = [-0.0855, -0.08906, -0.11692, -0.08501, -0.11355, -0.09287, 0.010086, 0.02994, 0.03919, 0.052103, 0.045846, 0.060084, 0.088319, 0.120182, 0.166933, 0.193817, 0.247547,
            0.306569, 0.349119, 0.384546, 0.343663, 0.331747, 0.357514]
        
        b2 = [-0.09256, -0.09162, -0.07838, -0.057, -0.07533, -0.10243, -0.10518, -0.12717, -0.13858, -0.15139, -0.16209, -0.1659, -0.16411, -0.16333, -0.16111, -0.15655, -0.15382,
            -0.14756, -0.14948, -0.13987, -0.13593, -0.14828, -0.12254]
        
        b3 = [0, 0, 0, 0, 0, 0.073904, 0.150461, 0.178899, 0.189682, 0.216011, 0.224827, 0.197716, 0.15475, 0.117576, 0.112005, 0.051729, 0.081575, 0.092837, 0.108209, 0.098737, 0,
            0, 0]
        
        y = [-0.3019, -0.29402, -0.24177, -0.20763, -0.17324, -0.20249, -0.29123, -0.35443, -0.39306, -0.45391, -0.49206, -0.56446, -0.5962, -0.66782, -0.73839, -0.79408, -0.8217,
            -0.82658, -0.84505, -0.8232, -0.77866, -0.76924, -0.76961] 

        sofN = [-0.03977, -0.03924, -0.03772, -0.04594, -0.03805, -0.02673, -0.03265, -0.03384, -0.03725, -0.02791, -0.02563, -0.01866, -0.01742, -0.00049, 0.011203, 0.016526,
                0.016449, 0.026307, 0.025234, 0.018674, 0.011371, 0.005535, 0.008735]
        
        sofR = [0.077525, 0.081052, 0.079778, 0.087497, 0.08471, 0.067844, 0.075977, 0.074982, 0.076701, 0.06979, 0.072567, 0.064599, 0.060283, 0.044921, 0.028151, 0.020352,
                0.021242, 0.018604, 0.022362, 0.023089, 0.016688, 0.019857, 0.023314 ]
        
        sofS = [-0.03776, -0.04182, -0.04206, -0.04155, -0.04666, -0.04111, -0.04332, -0.04114, -0.03946, -0.04188, -0.04694, -0.04594, -0.04286, -0.04443, -0.03935, -0.03688,
                -0.03769, -0.04491, -0.0476, -0.04176, -0.02806, -0.02539, -0.03205]

        sigma = [0.319753, 0.323885, 0.329654, 0.33886, 0.346379, 0.3419, 0.335532, 0.338114, 0.336741, 0.337694, 0.336278, 0.33929, 0.341717, 0.344388, 0.345788, 0.3452,
                 0.350517, 0.356067, 0.356504, 0.362835, 0.36502, 0.368857, 0.363037] 

    elif distance == 'Rhypo':
        e1 = [4.273, 4.3397, 4.46839, 4.5724, 4.55255, 4.51119, 4.49571, 4.492, 4.517, 4.4655, 4.468, 4.3715, 4.3419, 4.148, 4.092, 4.08324, 4.072, 3.779, 3.694, 3.454, 3.389,
              3.066]
        c1 = [-1.578, -1.604, -1.685, -1.638, -1.579, -1.447, -1.370, -1.366, -1.40, -1.41, -1.428, -1.40655, -1.397, -1.371, -1.377, -1.386, -1.387, -1.273, -1.264, -1.273,
              -1.282, -1.234]
        c2 = [0.108, 0.103, 0.126, 0.123, 0.125, 0.084, 0.038, 0.012, 0.00198, 0.000489, -0.0091, 0.00101, 0.004238, 0.00226, 0.008956, -0.00453, -0.01855, -0.01377, -0.00337,
              0.0837, 0.0867, 0.1501]
        h = [4.827, 4.47, 4.58, 5.12, 5.67, 4.82, 4.56, 3.94, 4.268, 4.399, 4.60, 4.60, 4.43, 3.009, 3.157, 3.45, 3.316, 3.049, 3.65, 4.60, 4.95, 4.45]
        
        c3 = [9.64e-5, 2.63e-5, 0., 0.000722, 0.001239, 0.001692, 0.001586, 0.001059, 0.000565, 5.97e-5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        
        b1 = [0.271, 0.23, 0.205, 0.226, 0.167, 0.194, 0.289, 0.321, 0.336, 0.346, 0.353, 0.3571, 0.3845, 0.466, 0.510, 0.567, 0.631, 0.65, 0.674, 0.563, 0.548, 0.541]
        
        b2 = [-0.068, -0.066, -0.0528, -0.0298, -0.050, -0.078, -0.08155, -0.10418, -0.11526, -0.12711, -0.1377, -0.1427, -0.1409, -0.138, -0.1326, -0.127, -0.1212, -0.129,
              -0.119, -0.1178, -0.1295, -0.1037]
        
        b3 = [0.3529, 0.369, 0.323, 0.311, 0.3498, 0.448, 0.53, 0.596, 0.612, 0.60, 0.621, 0.589, 0.543, 0.498, 0.437, 0.458, 0.474, 0.488, 0.461, 0.184, 0.171, 0.009]
        
        y = [-0.293, -0.286, -0.232, -0.195, -0.168, -0.1945, -0.2709, -0.323, -0.363, -0.43, -0.467, -0.5316, -0.555, -0.698, -0.757, -0.786, -0.791, -0.803, -0.7801, -0.749,
             -0.744, -0.744]
        
        sofN = [-0.047, -0.046, -0.045, -0.0532, -0.047, -0.036, -0.038, -0.036, -0.038, -0.028, -0.026, -0.019, -0.0175, 0.010, 0.015, 0.0163, 0.026, 0.024, 0.019, 0.011,
                0.004, 0.006]
        
        sofR = [0.11, 0.115, 0.1145, 0.1216, 0.119, 0.102, 0.107, 0.103, 0.104, 0.0955, 0.0971, 0.0902, 0.086, 0.054, 0.0458, 0.044, 0.0411, 0.0383, 0.038, 0.029, 0.033, 0.030]
        
        sofS = [-0.063, -0.068, -0.06943, -0.06845, -0.071, -0.066, -0.0688, -0.066, -0.06675, -0.06697, -0.071, -0.07092, -0.06843, -0.064, -0.060, -0.0606, -0.06753, -0.06325,
                -0.0578, -0.04092, -0.03858, -0.03653]
        
        sigma = [0.325, 0.3294, 0.337, 0.346, 0.352, 0.343, 0.341, 0.3507, 0.352, 0.3515, 0.352, 0.355, 0.358, 0.368, 0.374, 0.386, 0.398, 0.387, 0.393, 0.398, 0.4035, 0.389]
    
    # - Reference values
    M_ref = 5.5
    Mh = 6.75
    R_ref = 1.
    v_ref = 800. # - (m/s)
    
    # - Distance term
    Fd = (c1[ip] + c2[ip]*(M - M_ref))*math.log10((R**2. + h[ip]**2.)**0.5/R_ref) - c3[ip]*((R**2. + h[ip]**2.)**0.5-R_ref)
    
    # - Magnitude term
    if M <= Mh:
        Fm = b1[ip]*(M - Mh) + b2[ip]*(M - Mh)**2.
    else:
        Fm = b3[ip]*(M - Mh)
    
    # - Site effects term
    Fs = y[ip]*np.log10(vs30/v_ref)
    
    # - Style of faulting term
    if sof is 'S':
        Fsof = sofS[ip]
    elif sof is 'R':
        Fsof = sofR[ip]
    elif sof is 'N':
        Fsof = sofN[ip]
    
    log10Y = e1[ip] + Fd + Fm + Fs + Fsof
    Y = 10.**(log10Y)
    std = sigma[ip]
    return Y, std
# ----------------------------------------------------------------------------------------------------------------------- #
def AkkarRjb2014(M, T, Rjb, rake, vs30, stddev_type='Total', adjusting_factor=1.):
    """
    # S. Akkar, M. A. Sandikkaya, and J. J. Bommer
    #as published in "Empirical Ground-Motion Models for Point- and Extended-
    #Source Crustal Earthquake Scenarios in Europe and the Middle East",
    #Bulletin of Earthquake Engineering (2014), 12(1): 359 - 387
    #The class implements the equations for Joyner-Boore distance and based on
    #manuscript provided by the original authors.
    #
    # Provides ground-motion prediction equations for computing medians and
    # standard deviations of average horizontal components of PGA, PGV and 5#
    # damped linear pseudo-absolute aceleration response spectra for European
    #active shallow crust region
    #
    ###########################################################################
    # Input Variables
    # M             = Magnitude
    # T             = Period (sec);
    #                 Use 1000 for output the array of Sa with period
    # Rjb           = Joyner and Boore distance (to a maximum of 300 km)
    # vs30          = shear wave velocity averaged over top 30 m (m/s)
    # rake          = Rupture rake (indicates the type of rupture)
    # stddev_type   = Boolean indicating the type of standard deviation (Total, Intra, Inter)
    # adjusting_factor = Normally defined as 1.
    # 
    # Output Variables
    # Sa            = Median spectral acceleration prediction (ln(m/s))
    # sigma         = logarithmic standard deviation of spectral acceleration
    #                 prediction
    ###########################################################################
    """
    period = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24,
       0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
       0.95, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4, 3.6, 3.8, 4., -10.]
    n_periods = len(period) - 1
    
    # - Compute Sa and sigma with predefined periods
    if type(T) is int and T == 1000:
        Sa, Sigma = np.zeros(n_periods), np.zeros(n_periods)
        for ipT in range(n_periods):
            Sa[ipT], Sigma[ipT] = AkkarEtAlRjb2014_sub(ipT, M, Rjb, rake, vs30, stddev_type=stddev_type, adjusting_factor=adjusting_factor)
        return period[:-1], Sa, Sigma
    else:
        n_T = len(T)
        Sa, Sigma = np.zeros(n_T), np.zeros(n_T)
        for i in range(n_T):
            Ti = T[i]
            if Ti not in period: # The user defined period requires interpolation
                ip_low = np.where(period < Ti)[0][-1]
                ip_high = np.where(period > Ti)[0][0]
                T_low = period[ip_low]
                T_high = period[ip_high]
                
                Sa_low, Sigma_low = AkkarEtAlRjb2014_sub(ip_low, M, Rjb, rake, vs30, stddev_type=stddev_type, adjusting_factor=adjusting_factor)
                Sa_high, Sigma_high = AkkarEtAlRjb2014_sub(ip_high, M, Rjb, rake, vs30, stddev_type=stddev_type, adjusting_factor=adjusting_factor)
                x = [np.log(T_low), np.log(T_high)]
                Y_sa = [np.log(Sa_low), np.log(Sa_high)]
                Y_sigma = [np.log(Sigma_low), np.log(Sigma_high)]
                Sa[i] = np.exp(np.interp(np.log(Ti), x, Y_sa))
                Sigma[i] = np.exp(np.interp(np.log(Ti), x, Y_sigma))
            else:
                ip_T = np.where(period == Ti)[0][0]
                Sa[i], Sigma[i] = AkkarEtAlRjb2014_sub(ip_T, M, Rjb, rake, vs30, stddev_type=stddev_type, adjusting_factor=adjusting_factor)
        return Sa, Sigma
    

def AkkarEtAlRjb2014_sub(index, M, Rjb, rake, vs30, stddev_type='Total', adjusting_factor=1.):

    #: Coefficient table (from Table 3 and 4a, page 22)
    #: Table 4.a: Period-dependent regression coefficients of the RJB
    #: ground-motion model
    #: sigma is the 'intra-event' standard deviation, while tau is the
    #: 'inter-event' standard deviation

    a1 = [1.85329, 1.87032, 1.95279, 2.07006, 2.20452, 2.35413, 2.63078, 2.85412, 2.89772, 2.92748, 2.95162, 2.96299, 2.96622,
          2.93166, 2.88988, 2.84627, 2.79778, 2.73872, 2.63479, 2.53886, 2.48747, 2.38739, 2.3015, 2.17298, 2.07474, 2.01953,
          1.95078, 1.89372, 1.83717, 1.77528, 1.73155, 1.70132, 1.67127, 1.53838, 1.37505, 1.21156, 1.09262, 0.95211, 0.85227,
          0.76564, 0.66856, 0.58739, 0.52349, 0.3768, 0.23251, 0.10481, 0.00887, -0.01867, -0.0996, -0.21166, -0.273, -0.35366,
          -0.42891, -0.55307, -0.67806, -0.80494, -0.91278, -1.05642, -1.17715, -1.22091, -1.34547, -1.3979, -1.37536, 5.61201]
    a3 = [-0.02807, -0.0274, -0.02715, -0.02403, -0.01797, -0.01248, -0.00532, -0.00925, -0.01062, -0.01291, -0.01592, -0.01866,
          -0.02193, -0.02429, -0.02712, -0.03003, -0.033, -0.03462, -0.03789, -0.04173, -0.04768, -0.05178, -0.05672, -0.06015, -0.06508,
          -0.06974, -0.07346, -0.07684, -0.0801, -0.08296, -0.08623, -0.0907, -0.0949, -0.10275, -0.10747, -0.11262, -0.11835,
          -0.12347, -0.12678, -0.13133, -0.13551, -0.13957, -0.14345, -0.15051, -0.15527, -0.16106, -0.16654, -0.17187, -0.17728,
          -0.17908, -0.18438, -0.18741, -0.19029, -0.19683, -0.20339, -0.20703, -0.21074, -0.21392, -0.21361, -0.21951, -0.22724,
          -0.2318, -0.23848, -0.0998]
    a4 = [-1.23452, -1.23698, -1.25363, -1.27525, -1.30123, -1.32632, -1.35722, -1.38182, -1.38345, -1.37997, -1.37627, -1.37155,
          -1.3646, -1.35074, -1.33454, -1.31959, -1.3045, -1.28877, -1.26125, -1.236, -1.21882, -1.19543, -1.17072, -1.13847, -1.11131,
          -1.09484, -1.07812, -1.0653, -1.05451, -1.04332, -1.03572, -1.02724, -1.01909, -0.99351, -0.96429, -0.93347, -0.91162,
          -0.88393, -0.86884, -0.85442, -0.83929, -0.82668, -0.81838, -0.79691, -0.77813, -0.75888, -0.74871, -0.75751, -0.74823,
          -0.73766, -0.72996, -0.72279, -0.72033, -0.71662, -0.70452, -0.69691, -0.6956, -0.69085, -0.67711, -0.68177, -0.65918,
          -0.65298, -0.66482, -0.98388]
    a8 = [-0.1091, -0.1115, -0.104, -0.0973, -0.0884, -0.0853, -0.0779, -0.0749, -0.0704, -0.0604, -0.049, -0.0377, -0.0265,
          -0.0194, -0.0125, -0.0056, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0616]
    a9 = [0.0937, 0.0953, 0.1029, 0.1148, 0.1073 ,0.1052 , 0.0837, 0.0761, 0.0707, 0.0653, 0.0617, 0.0581, 0.0545, 0.0509, 0.0507,
          0.0502, 0.0497, 0.0493, 0.0488, 0.0483, 0.0478, 0.0474, 0.0469, 0.0464, 0.0459, 0.0459, 0.0429, 0.04, 0.0374, 0.0349, 0.0323,
          0.0297, 0.0271, 0.0245, 0.0219, 0.0193, 0.0167, 0.0141, 0.0115, 0.0089, 0.0062, 0.0016, 0, 0, 0, 0, 0, 0, 0, 0, -0.003, -0.006,
          -0.009, -0.0141, -0.0284, -0.0408, -0.0534, -0.0683, -0.078, -0.0943, -0.1278, -0.1744, -0.2231, 0.063]
    b1 = [-0.41997, -0.41729, -0.39998, -0.34799, -0.27572, -0.21231, -0.14427, -0.27064, -0.31025, -0.34796, -0.39668, -0.43996,
          -0.48313, -0.52431, -0.5568, -0.58922, -0.62635, -0.65315, -0.68711, -0.72744, -0.77335, -0.80508, -0.82609, -0.8408,
          -0.86251, -0.87479, -0.88522, -0.89517, -0.90875, -0.91922, -0.9267, -0.9372, -0.94614, -0.96564, -0.98499, -0.99733,
          -1.00469, -1.00786, -1.00606, -1.01093, -1.01576, -1.01353, -1.01331, -1.0124, -1.00489, -0.98876, -0.9776, -0.98071,
          -0.96369, -0.94634, -0.93606, -0.91408, -0.91007, -0.89376, -0.87052, -0.85889, -0.86106, -0.85793, -0.82094, -0.84449,
          -0.83216, -0.79216, -0.75645, -0.72057]
    b2 = [-0.28846, -0.28685, -0.28241, -0.26842, -0.24759, -0.22385, -0.17525, -0.29293, -0.31837, -0.3386, -0.36646, -0.38417, 
          -0.39551, -0.40869, -0.41528, -0.42717, -0.4413, -0.44644, -0.44872, -0.46341, -0.48705, -0.47334, -0.4573, -0.44267,
          -0.43888, -0.4382, -0.43678, -0.43008, -0.4219, -0.40903, -0.39442, -0.38462, -0.37408, -0.35582, -0.34053, -0.30949,
          -0.28772, -0.28957, -0.28555, -0.28364, -0.28037, -0.2839, -0.28702, -0.27669, -0.27538, -0.25008, -0.23508, -0.24695,
          -0.2287, -0.21655, -0.20302, -0.18228, -0.17336, -0.15463, -0.13181, -0.14066, -0.13882, -0.13336, -0.1377, -0.15337, 
          -0.10884, -0.08884, -0.07749, -0.19688]
    sigma = [0.6201, 0.6215, 0.6266, 0.641, 0.6534, 0.6622, 0.6626, 0.667, 0.6712, 0.6768, 0.6789, 0.6822, 0.6796, 0.6762, 0.6723,
             0.6694, 0.6647, 0.6645, 0.66, 0.6651, 0.665, 0.659, 0.6599, 0.6654, 0.6651, 0.6662, 0.6698, 0.6697, 0.6696, 0.6641, 
             0.6575, 0.654, 0.6512, 0.657, 0.663, 0.6652, 0.6696, 0.6744, 0.6716, 0.6713, 0.6738, 0.6767, 0.6787, 0.6912, 0.7015,
             0.7017, 0.7141, 0.7164, 0.7198, 0.7226, 0.7241, 0.7266, 0.7254, 0.7207, 0.7144, 0.7122, 0.7129, 0.6997, 0.682, 0.6682,
             0.6508, 0.6389, 0.6196, 0.6014]
    tau = [0.3501, 0.3526, 0.3555, 0.3565, 0.3484, 0.3551, 0.3759, 0.4067, 0.4059, 0.4022, 0.4017, 0.3945, 0.3893, 0.3928, 0.396,
           0.396, 0.3932, 0.3842, 0.3887, 0.3792, 0.3754, 0.3757, 0.3816, 0.3866, 0.3881, 0.3924, 0.3945, 0.3962, 0.389, 0.3929,
           0.4009, 0.4022, 0.4021, 0.4057, 0.406, 0.4124, 0.4135, 0.4043, 0.3974, 0.3971, 0.3986, 0.3949, 0.3943, 0.3806, 0.3802,
           0.3803, 0.3766, 0.3799, 0.3817, 0.3724, 0.371, 0.3745, 0.3717, 0.3758, 0.3973, 0.4001, 0.4025, 0.4046, 0.4194, 0.3971,
           0.4211, 0.415, 0.3566, 0.3311]
    
    """
    #: c1 is the reference magnitude, fixed to 6.75Mw (which happens to be the
    #: same value used in Boore and Atkinson, 2008)
    #: see paragraph 'Functional Form of Predictive Equations and Regressions',
    #: page 21
    """
    
    c1 = 6.75
    c = 2.5
    n = 3.2
    a2, a5, a6, a7 = 0.0029, 0.259, 7.5, 0.5096
    # -******************* METHODS *******************- #
    def _compute_mean(index, M, R, rake):
        """
        Compute and return mean value without site conditions,
        that is equations (1a) and (1b), p.2981-2982.
        """
        mean = (a1[index] + _compute_linear_magnitude_term(index, M) + _compute_quadratic_magnitude_term(index, M) +
                _compute_logarithmic_distance_term(index, M, R) + _compute_faulting_style_term(index, rake))

        return mean
    # ---------------------------------------------------------------------------------------------- #
    def _compute_non_linear_term(index, pga_only, vs30):
        """
        Compute non-linear term, equation (3a) to (3c), page 20.
        """
        Vref = 750.0
        Vcon = 1000.0
        if vs30 < Vref:
            # equation (3a)
            lnS = (b1[index] * np.log(vs30/Vref) + b2[index] * np.log((pga_only + c * (vs30/Vref) ** n) /
                ((pga_only + c) * (vs30/Vref) ** n)))
        elif vs30 >= Vref and vs30 <= Vcon:
            # equation (3b)
            lnS = b1[index] * np.log(vs30/Vref)
        elif vs30 > Vcon:
            # equation (3c)
            lnS = b1[index] * np.log(Vcon/Vref)

        return lnS
    # ---------------------------------------------------------------------------------------------- #
    def _compute_linear_magnitude_term(index, M):
        """
        Compute and return second term in equations (2a)
        and (2b), page 20.
        """
        if M <= c1:
            # this is the second term in eq. (2a), p. 20
            return a2 * (M - c1)
        else:
            # this is the second term in eq. (2b), p. 20
            return a7 * (M - c1)
    # ---------------------------------------------------------------------------------------------- #
    def _compute_faulting_style_term(index, rake):
        """
        Compute and return fifth and sixth terms in equations (2a)
        and (2b), pages 20.
        """
        Fn = float(rake > -135.0 and rake < -45.0)
        Fr = float(rake > 45.0 and rake < 135.0)

        return a8[index] * Fn + a9[index] * Fr
    # ---------------------------------------------------------------------------------------------- #
    def _compute_logarithmic_distance_term(index, M, Rjb):
        """
        Compute and return fourth term in equations (2a)
        and (2b), page 20.
        """
        return ((a4[index] + a5 * (M - c1)) * np.log(np.sqrt(Rjb**2. + a6**2.)))
    # ---------------------------------------------------------------------------------------------- #
    def _compute_quadratic_magnitude_term(index, M):
        """
        Compute and return third term in equations (2a)
        and (2b), page  20.
        """
        return a3[index] * (8.5 - M) ** 2.
    # ---------------------------------------------------------------------------------------------- #
    def _get_stddevs(index):
        """
        Return standard deviations as defined in table 4a, p. 22.
        """
        if stddev_type is 'Total':
            Sigma = np.sqrt(sigma[index] ** 2. + tau[index] ** 2.)
        elif stddev_type is 'Intra':
            Sigma = sigma[index]
        elif stddev_type is 'Inter':
            Sigma = tau[index]
        else:
            print('Select a valid type of standard deviation')
        return Sigma
    # ---------------------------------------------------------------------------------------------- #
    
    # compute median PGA on rock, needed to compute non-linear site amplification
    median_pga = np.exp(_compute_mean(index, M, Rjb, rake))
    
    # compute full mean value by adding nonlinear site amplification terms    
    mean = (_compute_mean(index, M, Rjb, rake) + _compute_non_linear_term(index, median_pga, vs30))
    stddevs = _get_stddevs(index)

    return mean, stddevs

# ----------------------------------------------------------------------------------------------------------------------- #
def AkkarRhypo2014(M, T, Rhypo, rake, vs30, stddev_type='Total'):
    """
    # S. Akkar, M. A. Sandikkaya, and J. J. Bommer
    #as published in "Empirical Ground-Motion Models for Point- and Extended-
    #Source Crustal Earthquake Scenarios in Europe and the Middle East",
    #Bulletin of Earthquake Engineering (2014), 12(1): 359 - 387
    #The class implements the equations for Joyner-Boore distance and based on
    #manuscript provided by the original authors.
    #
    # Provides ground-motion prediction equations for computing medians and
    # standard deviations of average horizontal components of PGA, PGV and 5#
    # damped linear pseudo-absolute aceleration response spectra for European
    #active shallow crust region
    #
    ###########################################################################
    # Input Variables
    # M             = Magnitude
    # T             = Period (sec);
    #                 Use 1000 for output the array of Sa with period
    # Rhypo          = Hypocentral distance (to a maximum of 300 km)
    # vs30          = shear wave velocity averaged over top 30 m (m/s)
    # rake          = Rupture rake (indicates the type of rupture)
    # stddev_type   = Boolean indicating the type of standard deviation (Total, Intra, Inter)
    # adjusting_factor = Normally defined as 1.
    # 
    # Output Variables
    # Sa            = Median spectral acceleration prediction (ln(m/s))
    # sigma         = logarithmic standard deviation of spectral acceleration
    #                 prediction
    ###########################################################################
    """
    period = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.22, 0.24,
       0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
       0.95, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4, 3.6, 3.8, 4., -10.]
    n_periods = len(period) - 1
    
    # - Compute Sa and sigma with predefined periods
    if type(T) is int and T == 1000:
        Sa, Sigma = np.zeros(n_periods), np.zeros(n_periods)
        for ipT in range(n_periods):
            Sa[ipT], Sigma[ipT] = AkkarEtAlRhypo2014_sub(ipT, M, Rhypo, rake, vs30, stddev_type=stddev_type)
        return period[:-1], Sa, Sigma
    else:
        n_T = len(T)
        Sa, Sigma = np.zeros(n_T), np.zeros(n_T)
        for i in range(n_T):
            Ti = T[i]
            if Ti not in period: # The user defined period requires interpolation
                ip_low = np.where(period < Ti)[0][-1]
                ip_high = np.where(period > Ti)[0][0]
                T_low = period[ip_low]
                T_high = period[ip_high]
                
                Sa_low, Sigma_low = AkkarEtAlRhypo2014_sub(ip_low, M, Rhypo, rake, vs30, stddev_type=stddev_type)
                Sa_high, Sigma_high = AkkarEtAlRhypo2014_sub(ip_high, M, Rhypo, rake, vs30, stddev_type=stddev_type)
                x = [np.log(T_low), np.log(T_high)]
                Y_sa = [np.log(Sa_low), np.log(Sa_high)]
                Y_sigma = [np.log(Sigma_low), np.log(Sigma_high)]
                Sa[i] = np.exp(np.interp(np.log(Ti), x, Y_sa))
                Sigma[i] = np.exp(np.interp(np.log(Ti), x, Y_sigma))
            else:
                ip_T = np.where(period == Ti)[0][0]
                Sa[i], Sigma[i] = AkkarEtAlRhypo2014_sub(ip_T, M, Rhypo, rake, vs30, stddev_type=stddev_type)
        return Sa, Sigma

def AkkarEtAlRhypo2014_sub(index, M, Rhypo, rake, vs30, stddev_type='Total'):

    #: Coefficient table (from Table 3 and 4a, page 22)
    #: Table 4.a: Period-dependent regression coefficients of the RJB
    #: ground-motion model
    #: sigma is the 'intra-event' standard deviation, while tau is the
    #: 'inter-event' standard deviation
    
    a1 = [3.26685, 3.28656, 3.38936, 3.53155, 3.68895, 3.86581, 4.18224, 4.4375, 4.48828, 4.51414, 4.5329, 4.53834, 4.52949, 4.47016, 4.40011, 4.33238, 4.26395, 4.1775,
        4.03111, 3.90131, 3.82611, 3.6978, 3.57698, 3.40759, 3.2758, 3.19725, 3.11035, 3.03752, 2.97485, 2.90617, 2.85484, 2.8172, 2.77997, 2.62299, 2.42234, 2.2277, 
        2.08102, 1.91625, 1.81167, 1.71853, 1.60822, 1.51532, 1.43982, 1.26728, 1.11475, 0.95965, 0.85203, 0.83007, 0.74487, 0.63568, 0.56996, 0.485, 0.40614, 0.28608,
        0.15432, 0.0225, -0.07822, -0.22534, -0.36165, -0.39423, -0.54126, -0.59607, -0.51893]

    a3 = [-0.04846, -0.04784, -0.04796, -0.04537, -0.03991, -0.0349, -0.02826, -0.03256, -0.03407, -0.03635, -0.03929, -0.042, -0.04509, -0.04701, -0.04932, -0.05181, -0.05442,
        -0.05565, -0.05817, -0.06152, -0.06706, -0.0706, -0.0749, -0.07756, -0.08183, -0.08602, -0.08937, -0.09243, -0.09556, -0.09822, -0.10132, -0.1056, -0.10964, -0.11701,
        -0.12106, -0.12555, -0.13074, -0.13547, -0.13856, -0.14294, -0.14669, -0.15056, -0.15427, -0.16107, -0.1663, -0.1717, -0.17699, -0.18248, -0.18787, -0.18961, -0.19551,
        -0.19853, -0.20136, -0.20791, -0.2148, -0.21843, -0.22224, -0.22564, -0.22496, -0.23237, -0.24003, -0.24448, -0.25256]

    a4 = [-1.47905, -1.48197, -1.50214, -1.52781, -1.55693, -1.58672, -1.62527, -1.65601, -1.65903, -1.6547, -1.64994, -1.64398, -1.63467, -1.61626, -1.59485, -1.57545,
        -1.55685, -1.53574, -1.50045, -1.46889, -1.44738, -1.41925, -1.38832, -1.34898, -1.31609, -1.29558, -1.27591, -1.26045, -1.24891, -1.237, -1.22822, -1.21874, -1.20953,
        -1.1801, -1.14424, -1.10853, -1.08192, -1.05027, -1.03514, -1.0201, -1.00315, -0.98859, -0.97812, -0.95163, -0.93048, -0.90604, -0.89379, -0.90319, -0.89323, -0.88392,
        -0.87459, -0.86659, -0.86343, -0.86086, -0.84778, -0.83937, -0.83964, -0.83314, -0.81702, -0.82109, -0.79431, -0.78785, -0.80922]

    a8 = [-0.1091, -0.1115, -0.104, -0.0973, -0.0884, -0.0853, -0.0779, -0.0749, -0.0704, -0.0604, -0.049, -0.0377, -0.0265, -0.0194, -0.0125, -0.0056, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    a9 = [0.0937, 0.0953, 0.1029, 0.1148, 0.1073, 0.1052, 0.0837, 0.0761, 0.0707, 0.0653, 0.0617, 0.0581, 0.0545, 0.0509, 0.0507, 0.0502, 0.0497, 0.0493, 0.0488, 0.0483, 0.0478,
        0.0474, 0.0469, 0.0464, 0.0459, 0.0459, 0.0429, 0.04, 0.0374, 0.0349, 0.0323, 0.0297, 0.0271, 0.0245, 0.0219, 0.0193, 0.0167, 0.0141, 0.0115, 0.0089, 0.0062, 0.0016,
        0, 0, 0, 0, 0, 0, 0, 0, -0.003, -0.006, -0.009, -0.0141, -0.0284, -0.0408, -0.0534, -0.0683, -0.078, -0.0943, -0.1278, -0.1744, -0.2231]

    b1 = [-0.41997, -0.41729, -0.39998, -0.34799, -0.27572, -0.21231, -0.14427, -0.27064, -0.31025, -0.34796, -0.39668, -0.43996, -0.48313, -0.52431,
          -0.5568, -0.58922, -0.62635, -0.65315, -0.68711, -0.72744, -0.77335, -0.80508, -0.82609, -0.8408, -0.86251, -0.87479, -0.88522, -0.89517,
          -0.90875, -0.91922, -0.9267, -0.9372, -0.94614, -0.96564, -0.98499, -0.99733, -1.00469, -1.00786, -1.00606, -1.01093, -1.01576, -1.01353,
          -1.01331, -1.0124, -1.00489, -0.98876, -0.9776, -0.98071, -0.96369, -0.94634, -0.93606, -0.91408,-0.91007, -0.89376, -0.87052, -0.85889,
          -0.86106, -0.85793, -0.82094, -0.84449, -0.83216, -0.79216, -0.75645]
    
    b2 = [-0.28846, -0.28685, -0.28241, -0.26842, -0.24759, -0.22385, -0.17525, -0.29293, -0.31837, -0.3386, -0.36646, -0.38417, -0.39551, -0.40869,
          -0.41528, -0.42717, -0.4413, -0.44644, -0.44872, -0.46341, -0.48705, -0.47334, -0.4573, -0.44267, -0.43888, -0.4382, -0.43678, -0.43008, 
          -0.4219, -0.40903, -0.39442, -0.38462, -0.37408, -0.35582, -0.34053, -0.30949, -0.28772, -0.28957, -0.28555, -0.28364, -0.28037, -0.2839,
          -0.28702, -0.27669, -0.27538, -0.25008, -0.23508, -0.24695, -0.2287, -0.21655, -0.20302, -0.18228, -0.17336, -0.15463, -0.13181, -0.14066,
          -0.13882, -0.13336, -0.1377, -0.15337, -0.10884, -0.08884, -0.07749]

    sigma = [0.6475, 0.6492, 0.6543, 0.6685, 0.6816, 0.6899, 0.6881, 0.6936, 0.6965, 0.7022, 0.7043, 0.7071, 0.7048, 0.7032, 0.7011, 0.6992, 0.6947, 0.6954, 0.6925, 0.6973,
            0.6973, 0.6914, 0.6934, 0.6992, 0.699, 0.7006, 0.7036, 0.7037, 0.7023, 0.6956, 0.6893, 0.6852, 0.6821, 0.6866, 0.6926, 0.6949, 0.6993, 0.7028, 0.6981, 0.6959,
            0.6983, 0.7006, 0.7022, 0.7137, 0.7224, 0.7226, 0.7349, 0.7378, 0.7406, 0.7418, 0.7431, 0.7457, 0.7446, 0.7391, 0.7311, 0.7281, 0.7279, 0.7154, 0.6984, 0.6867,
            0.6687, 0.6565, 0.6364]

    tau = [0.3472, 0.3481, 0.3508, 0.3526, 0.3513, 0.3659, 0.3942, 0.4122, 0.4065, 0.3964, 0.3937, 0.3853, 0.3779, 0.3851, 0.39, 0.3889, 0.3903, 0.3848, 0.3891, 0.3839,
        0.3839, 0.3865, 0.3896, 0.3908, 0.3888, 0.3916, 0.3913, 0.3894, 0.3847, 0.3908, 0.3986, 0.4017, 0.4017, 0.4044, 0.4005, 0.3981, 0.3967, 0.389, 0.3824, 0.3831, 0.3825,
        0.3797, 0.3826, 0.3721, 0.3723, 0.3746, 0.3697, 0.3758, 0.3794, 0.3686, 0.3692, 0.3705, 0.3676, 0.3718, 0.3941, 0.3967, 0.3987, 0.4019, 0.4113, 0.38, 0.4009, 0.3952,
        0.3318]

    
    """
    #: c1 is the reference magnitude, fixed to 6.75Mw (which happens to be the
    #: same value used in Boore and Atkinson, 2008)
    #: see paragraph 'Functional Form of Predictive Equations and Regressions',
    #: page 21
    """
    a2 = 0.0029
    a5 = 0.25
    a6 = 7.5
    a7 = -0.5096
    c1 = 6.75
    Vcon = 1000.
    Vref = 750.
    c = 2.5
    n = 3.2
    # -******************* METHODS *******************- #
    def _compute_mean(index, M, R, rake):
        """
        Compute and return mean value without site conditions,
        that is equations (1a) and (1b), p.2981-2982.
        """
        mean = (a1[index] + _compute_linear_magnitude_term(index, M) + _compute_quadratic_magnitude_term(index, M) +
                _compute_logarithmic_distance_term(index, M, R) + _compute_faulting_style_term(index, rake))

        return mean
    # ---------------------------------------------------------------------------------------------- #
    def _compute_non_linear_term(index, pga_only, vs30):
        """
        Compute non-linear term, equation (3a) to (3c), page 20.
        """
        if vs30 < Vref:
        # equation (3a)
            lnS = (b1[index] * np.log(vs30/Vref) + b2[index] * np.log((pga_only + c * (vs30/Vref) ** n) /
                ((pga_only + c) * (vs30/Vref) ** n)))
        elif vs30 >= Vref and vs30 <= Vcon:
            # equation (3b)
            lnS = b1[index] * np.log(vs30/Vref)
        elif vs30 > Vcon:
            # equation (3c)
            lnS = b1[index] * np.log(Vcon/Vref)

        return lnS
    # ---------------------------------------------------------------------------------------------- #
    def _compute_linear_magnitude_term(index, M):
        """
        Compute and return second term in equations (2a)
        and (2b), page 20.
        """
        if M <= c1:
            # this is the second term in eq. (2a), p. 20
            return a2 * (M - c1)
        else:
            # this is the second term in eq. (2b), p. 20
            return a7 * (M - c1)
    # ------------------------------------------------------
    def _compute_faulting_style_term(index, rake):
        """
        Compute and return fifth and sixth terms in equations (2a)
        and (2b), pages 20.
        """
        Fn = float(rake > -135.0 and rake < -45.0)
        Fr = float(rake > 45.0 and rake < 135.0)

        return a8[index] * Fn + a9[index] * Fr
    # ---------------------------------------------------------------------------------------------- #
    def _compute_logarithmic_distance_term(index, M, Rhypo):
        """
        Compute and return fourth term in equations (2a)
        and (2b), page 20.
        """
        return ((a4[index] + a5 * (M - c1)) * np.log(np.sqrt(Rhypo**2. + a6**2.)))
    # ---------------------------------------------------------------------------------------------- #
    def _compute_quadratic_magnitude_term(index, M):
        """
        Compute and return third term in equations (2a)
        and (2b), page  20.
        """
        return a3[index] * (8.5 - M) ** 2.
    # ---------------------------------------------------------------------------------------------- #
    def _get_stddevs(index):
        """
        Return standard deviations as defined in table 4a, p. 22.
        """
        if stddev_type is 'Total':
            Sigma = np.sqrt(sigma[index] ** 2. + tau[index] ** 2.)
        elif stddev_type is 'Intra':
            Sigma = sigma[index]
        elif stddev_type is 'Inter':
            Sigma = tau[index]
        else:
            print('Select a valid type of standard deviation')
        return Sigma
    # ---------------------------------------------------------------------------------------------- #
    
    # compute median PGA on rock, needed to compute non-linear site amplification
    median_pga = np.exp(_compute_mean(index, M, Rhypo, rake))
    
    # compute full mean value by adding nonlinear site amplification terms    
    mean = (_compute_mean(index, M, Rhypo, rake) + _compute_non_linear_term(index, median_pga, vs30))
    stddevs = _get_stddevs(index)

    return mean, stddevs








