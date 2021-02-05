"""
Main objective of this program is to read the information from the pseudo-dynamic
rupture generation as described by Gallovic.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt

class gsource:
    def __init__(self, path):
        slip_data = np.loadtxt(path + '/' + 'slipdistribution.dat')
        #rupture_data = np.loadtxt(path + '/' + 'stf.dat')

        # - Coordinates 
        self.x, self.z = slip_data[:,0], slip_data[:,1]
        self.dx = self.x.max()/self.x.shape[0]**0.5
        self.dz = self.z.max()/self.z.shape[0]**0.5
        sub_sources_centroid = []
        for i in range(self.x.shape[0]):
            sub_sources_centroid.append([self.x[i], 0., self.z[i]])
        self.sub_sources_centroid = np.array(sub_sources_centroid)
        
        # - Slip distribution 
        self.slip = slip_data[:,2]
        
        # - Seismic moment
        self.mo = slip_data[:,3]
        
        # - Rupture time and activation order
        self.rupt_t = slip_data[:,4]
        rupture_ordered = copy.deepcopy(self.rupt_t)
        rupture_ordered.sort()
        integers = list(range(1, self.x.shape[0] + 1))
        
        """
        Sort the subfaults based on their rupture time.  This step is essential for the 
        dynamic corner frequency computation.
        """
        self.activation  = np.zeros(self.x.shape[0])
        for i in range(self.x.shape[0]):
            time_rupture = self.rupt_t[i]
            index = np.where(rupture_ordered == time_rupture)[0][0]
            self.activation[i] = integers[index]

        # - Stress drop
        self.sd = slip_data[:,5]
        
        # - Nucleation point
        hyp_index = np.where(self.activation == 1)[0][0]
        self.hypocenter = [self.sub_sources_centroid[hyp_index][0], 0., self.sub_sources_centroid[hyp_index][2]]

        
    def plotmap(self,variable, hypocenter=None, title=None):
        plt.figure(figsize=(10, 8))
        if title is not None:
            plt.title(title, fontsize=16)
            
        sc = plt.tricontourf(self.x, self.z, variable, cmap='binary')
        if hypocenter is not None:
            plt.scatter(hypocenter[0], hypocenter[1], marker='*', s=200, color='C1')
        plt.colorbar(sc, shrink=0.3)
        plt.xlabel('Distance along the strike [km]', fontsize=16)
        plt.ylabel('Distance down-dip [km]', fontsize=16)
        plt.tick_params(axis="x", labelsize=14)
        plt.tick_params(axis="y", labelsize=14)
        plt.xlim([self.x.min(), self.x.max()])
        plt.ylim([self.z.min(), self.z.max()])
        plt.show()
        plt.close()

    def plot_rupture_evol(self):
        import seaborn as sns
        plt.figure()
        plt.subplots_adjust(hspace=0.3)
        plt.subplot(211)
        plt.plot(self.tr, self.mr_tr)
        plt.ylabel("Moment Rate")

        plt.subplot(212)
        plt.plot(self.tr[:len(self.mo_tr)], self.mo_tr)
        plt.xlabel("Time [s]")
        plt.ylabel("Moment")
        sns.despine()
        plt.show()

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
            o =  (theta + 90.)*pi/180.  # - Strike is considered clock-wise
            R = np.array([[np.cos(o),-np.sin(o), 0], [np.sin(o), np.cos(o), 0], [0, 0, 1.]])
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
            o = (360. - (theta-90.))*pi/180.
            R = np.array([[1., 0, 0], [0., np.cos(o), -np.sin(o)],[0., np.sin(o), np.cos(o)]])
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

    
    
