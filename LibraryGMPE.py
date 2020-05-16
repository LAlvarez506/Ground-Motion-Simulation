import numpy as np

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
        
        return Sa, sigma, period
    
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
        

def BSSA_2014_nga(M, T, Rjb, Fault_Type, region, z1, Vs30):
    # coded by Yue Hua
#               Stanford University
#               yuehua@stanford.edu
#
# modified by Jack Baker, 3/3/2014
#
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
#
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
    
    
    if T == 1000:
        Sa = np.zeros(n_period)
        sigma = np.zeros(n_period)
        for ip in range(n_period):
            Sa[ip], sigma[ip] = BSSA_2014_sub(M, ip, Rjb, U, SS, NS, RS, region, z1, Vs30)
        return period, Sa, sigma
    else:
        n_T = len(T)
        Sa = np.zeros(n_T)
        sigma = np.zeros(n_T)
        for i in range(len(T)):
            Ti = T[i]
            if Ti not in period:
                index_lower = np.where(period < T)[0][-1]
                index_higher = np.where(period > T)[0][0]
                T_lo = period[index_lower]
                T_hi = period[index_higher]
                
                Sa_lo, sigma_lo = BSSA_2014_sub(M, index_lower, Rjb, U, SS, NS, RS, region, z1, Vs30)
                Sa_hi, sigma_hi = BSSA_2014_sub(M, index_higher, Rjb, U, SS, NS, RS, region, z1, Vs30)
                
                X = [T_lo, T_hi]
                Y_Sa = [Sa_lo, Sa_hi]
                Y_sigma = [sigma_lo, sigma_hi]
                Sa[i] = np.interp(T,X,Y_Sa)
                sigma[i] = np.interp(T,X,Y_sigma)
            else:
                index = np.where(period == T)[0][0]
                Sa[i], sigma[i] = BSSA_2014_sub(M, index, Rjb, U, SS, NS, RS, region, z1, Vs30)
            return Sa, sigma

def BSSA_2014_sub (M, ip, Rjb, U, SS, NS, RS, region, z1, Vs30):
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

    sigma = np.sqrt(phi_MRV**2. + tau**2.)
    
    return median, sigma
