import numpy as np
import matplotlib.pylab as plt

class source:
    def __init__(self):
        pass

    def create_plane(self,X,Z,nx,nz):
        self.corners = np.array([[0.0,0.0,Z],[X,0.0,Z],[0.0,0.0,0.0],[X,0.0,0.0]])
        
        # - Creates the vectors for the mesh
        self.x = np.linspace(0,X,nx+1)
        self.z = np.linspace(0,Z,nz+1)
        dx = self.x[1] - self.x[0]
        dz = self.z[1] - self.z[0]
        self.sub_source_area = dx*dz
        
        # - Find the coordinates of the points
        sub_sources_centroid = []
        for i in range(0,len(self.x)-1):
            for j in range(0,len(self.z)-1):
                xi,xf = self.x[i],self.x[i+1]
                zi,zf = self.z[j],self.z[j+1]
                
                xm = (xi+xf)*0.5
                zm = (zi+zf)*0.5
                sub_sources_centroid.append([xm,0.0,zm])
                
        self.sub_sources_centroid = np.array(sub_sources_centroid)

    def rupture_time_activation(self,v_rupture,hypocenter):
        """
        Computes the time required for the rupture to reach the centroid of each subfault,
        it also computes the activation order of each subfault
        """
        self.t_rupture,index = [],[]
        for i in range(len(self.sub_sources_centroid)):
            v = self.sub_sources_centroid[i] - np.array(hypocenter)
            d_rup = np.linalg.norm(v)
            self.t_rupture.append(d_rup/v_rupture)
            index.append(i)
            
        self.activation_index = [x for _,x in sorted(zip(self.t_rupture,index))]
            
        

    def rotate_plane(self,po,theta,axis):
        pi = np.pi
        """
        Initially, the rectangular fault is drawn in the X-Z plane.  Rotation is performed by means
        of basic rotational matrices.  Initially rotation in the X-axis is performed to provide the fault 
        with the dipping angle, where a dipping angle between 0-90 will produce a rotation counter-clockwise generating
        positive y coordinates for the points.
        
        Finally, the fault plane is rotated around the Z- axis so as to align it with the strike.  A positive angle rotates the plane counter clock-wise,
        for example if aligned in the x-axis, an leaving one end fixed, a positive angle would rotate towards the positive Y- axis.
        
        The rotations herein presented are with respect to the origin
        """
        def Rz(theta,M):
            """
            Strike is defined as an azimuth, with respect to the north and therefore,
            according to the convention of the code to the Y- axis
            """
            M = np.array(M)
            n = M.shape[0]
            o = (360.0 - (theta + 90.0))*pi/180.0  # - Strike is considered clock-wise
            R = np.array([[np.cos(o),-1*np.sin(o),0],[np.sin(o),np.cos(o),0],[0,0,1]])
            if len(M.shape) > 1:
                Mn = []
                for i in range(n):
                    pn = R.dot(M[i])
                    Mn.append(pn)
                Mn = np.array(Mn)
            else: 
                pn = R.dot(M)
                Mn = pn
            return Mn

        def Rx(theta,M):
            """
            dip is defined starting from level 0 (ground)
            """
            M = np.array(M)
            n = M.shape[0]
            o = (360.0-(theta-90.0))*pi/180.0
            R = np.array([[1,0,0],[0.0,np.cos(o),-1*np.sin(o)],[0.0,np.sin(o),np.cos(o)]])
            if len(M.shape) > 1:
                Mn = []
                for i in range(n):
                    pn = R.dot(M[i])
                    Mn.append(pn)
                Mn = np.array(Mn)
            else: 
                pn = R.dot(M)
                Mn = pn
            return Mn
        
        if axis is 'z':
            pn = Rz(theta,po)
        elif axis is 'x':
            pn = Rx(theta,po)

        return pn

    
    def generate_slip(self,X,Z,nx,nz,stochastic_field=None):
        """
        Creates a correlated random field based on a correlation 2 matrix.  Correlation is by means of a convulution of a normally
        distributed random sampling and a correlation matrix; FFT is used to pass from the spatial to frequencial domain
        
        X: Length of the faul
        Z: Depth of the fault
        nx,nz: discretization of the fault dimensions
        """
        
        correlation_type = 'vonkarman'
        a_x, a_z, H = 5., 2., 0.5
        slip_mean, slip_cov = 256., 0.8
        
        if stochastic_field is not None:
            if 'seed' in stochastic_field:
                np.random.seed(stochastic_field['seed'])
            if 'correlation_type' in stochastic_field:
                correlation_type = stochastic_field['correlation_type']
            if 'Mw' in stochastic_field:
                a_x = 10.0**(-2.5 + 0.5*stochastic_field['Mw'])
                a_z = 10.0**(-1.5 + (1.0/3.)*stochastic_field['Mw'])
            if 'H' in stochastic_field:
                H = stochastic_field['H']
            if 'slip_mean' in stochastic_field:
                slip_mean = stochastic_field['slip_mean']
            if 'slip_cov' in stochastic_field:
                slip_cov = stochastic_field['slip_cov']
        
        slip_std = slip_mean*slip_cov

        dx,dz = X/nx,Z/nz 
        
        # - Computes the length based on the discretization created from the final
        # number of points
        length_x = nx*dx # domain length (X direction)
        length_z = nz*dz# domain width (Z direction)
        
        ### - Prepare the Fourier domain for the 2d transform
        dkx = 2.*np.pi/length_x 
        dkz = 2.*np.pi/length_z 

        # - Computes the discretization of the wavenumbers
        kx1 = np.arange(0,nx/2 + 1)*dkx
        kx1 = np.hstack((kx1,np.flipud(kx1[1:int(nx/2)])))
        kz1 = np.arange(0,nz/2 + 1)*dkz
        kz1 = np.hstack((kz1,np.flipud(kz1[1:int(nz/2)])))
        [kx,kz] = np.meshgrid(kx1, kz1, indexing='ij')
        k = np.sqrt(kx**2.*a_x**2. + kz**2.*a_z**2.) # wavenumber matrix
        
        # - Correlation matrix the rnormal field
        self.random_data,phase_information = self.random_slip(nx,nz)
        

        # -Builds the correlation matrix for the stochastic slip-field

        if correlation_type is 'gaussian':
            auto_correl = (length_x*length_z*np.pi*a_x*a_z)*np.exp(-(k**2/4)) # PSD of autocorrelation function
            auto_correl = np.sqrt(auto_correl)  # PSD to Fourier transform, Fourier transform is square root of PSD

        if correlation_type is 'exponential':
            auto_correl = (length_x*length_z*np.pi*a_x*a_z)/(1+k**2)**1.5
            auto_correl = np.sqrt(auto_correl)  # PSD to Fourier transform, Fourier transform is square root of PSD

        if correlation_type is 'vonkarman':
            auto_correl = (length_x*length_z*np.pi*a_x*a_z)/(1+k**2)**(H+1)
            auto_correl = np.sqrt(auto_correl)  # PSD to Fourier transform, Fourier transform is square root of PSD
        
        
        wavenumber_product = auto_correl*np.exp((1j)*phase_information)
        spatial_correlated = np.real((np.fft.ifft2(wavenumber_product)))*nx*nz
        spatial_correlated = spatial_correlated * dkx *dkz /((2*np.pi)**2)
    
        # - Scale to median slip distribution 
        slip_dist = slip_mean + slip_std * spatial_correlated # transforming and stretching the output
        # - Check for zero or negative slips
        self.slip_dist = np.maximum(1e-5, slip_dist) # limiting the minimum to zero

    
    def random_slip(self,nx,nz):
        """
        Generate the random field
        """
        random_data = np.random.normal(0, 1, (nx,nz)) # initial pdf generated by normal distribution 
        random_data_transformed = (np.fft.fft2(random_data)) # transforming to wavenumber domain
        phase_information = np.angle(random_data_transformed) # getting phase information
        return random_data, phase_information
        
        
    def show_slip_dist(self,data):
        """
        Generate the random field
        """
        fig = plt.figure(figsize=(15, 10))
        axes = plt.gca()
        im = axes.imshow(np.transpose(data), cmap='jet', aspect='equal', extent=[min(self.x),max(self.x),min(self.z),max(self.z)], origin='lower')
        plt.rc('font', family='serif')
        axes.tick_params(direction='out', length=5, width=2,labelsize=16)
        plt.xlabel('X [km]', fontsize=18)
        plt.ylabel('Z [km]', fontsize=18)
        axes.xaxis.labelpad = 22
        axes.yaxis.labelpad = 22
        fig.colorbar(im,shrink=0.3)
        plt.show()
        plt.close()
        

    
