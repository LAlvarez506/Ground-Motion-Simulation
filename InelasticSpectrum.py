import numpy as np
import matplotlib.pyplot
pi = np.pi
"""
Computes the inelastic spectrum based on the finite differences method
"""
    
class inelastic_spectrum:

    def __init__(self):
        pass
    ################################################################################    
    def elastic_spectrum(self,periods,acc,dt,damping):
        """
        Finite differences resolution of the movement differential equation.  Not reliable for 
        periods below 0.05s due to numerical instability of the method
        """
        SD = []
        SV = []
        SA = []
        stiffness = []
        omega = []
        
        m = 1.0
        n = len(acc)
        for pp in range(len(periods)):
            # - Define aid variables     
            if periods[pp] == 0.0:
                p = 5e-2
            else: 
                p = periods[pp]
            
            w = 2*pi/p
            omega.append(w)
                
            k = m*(w**2)
            stiffness.append(k)
            c = 2*damping*(k*m)**0.5
            
            # - Initial conditions for Central differences
            u = []
            u0 = 0.0
            a0 = -1*acc[0]
            u1 = u0 + 0.5*a0*dt**2
            u.append(u0)
            u.append(u1)
            
            # - Run central differences
            for i in range(1,n-1):
                u_next = self.disp_next(u[i],u[i-1],acc[i],k,c,m,dt)
                u.append(u_next)
                
            
            sd = max(max(u),abs(min(u)))
            SD.append(sd)
            
        SD = np.array(SD)
        omega = np.array(omega)
        stiffness = np.array(stiffness)
        SV = omega*SD
        SA = SD*omega**2
        F_elastic = stiffness*SD
        return SA,SV,SD,F_elastic
    ################################################################################
    def inelastic_spectrum_Ry(self,periods,acc,dt,damping,Ry,alpha=None):
        
        SA,SV,SD,F_elastic = self.elastic_spectrum(periods,acc,dt,damping)
        
        SD_nl,SV_nl,SA_nl,nu,omega = [],[],[],[],[]
        m = 1.0
        n = len(acc)
        for pp in range(len(periods)):
            if periods[pp] == 0.0:
                p = 5e-2
            else: 
                p = periods[pp]
            
            u_e = SD[pp]
            f_e = F_elastic[pp]
            u_y = u_e/Ry
            f_y = f_e/Ry
    
            w = 2*pi/p
            omega.append(w)
                
            k = m*(w**2)
            c = 2*damping*(k*m)**0.5
            
            # - Initial conditions for Central differences
            u = []
            u0 = 0.0
            a0 = -1*acc[0]
            u1 = u0 + 0.5*a0*dt**2
            u.append(u0)
            u.append(u1)
            F = []
            F0 = 0.0
            F1 = u1*k
            F.append(F0)
            F.append(F1)

            # - Run central differences
            for i in range(1,n-1):
                u_next = self.disp_next(u[i],u[i-1],acc[i],k,c,m,dt,F_i = F[i])
                F_next = self.force_next(u_next,u[i],u[i-1],k,F[i],f_y,alpha=alpha)
                u.append(u_next)
                F.append(F_next)

            sd = max(max(u),abs(min(u)))
            SD_nl.append(sd)
            
            # - Ductility computation 
            ductility = sd/u_y
            nu.append(ductility)
        
        SD_nl = np.array(SD_nl)
        omega = np.array(omega)
        SV_nl = omega*SD_nl
        SA_nl = SD_nl*omega**2
        return SA_nl,SV_nl,SD_nl,nu,SA,SV,SD
    ################################################################################
    def Thotong2006(self,periods,acc,dt,damping,sde,Ry,alpha=None):

        SD_nl,SV_nl,SA_nl,nu,omega = [],[],[],[],[]
        m = 1.0
        n = len(acc)
        for pp in range(len(periods)):
            if periods[pp] == 0.0:
                p = 5e-2
            else: 
                p = periods[pp]
    
            w = 2*pi/p
            omega.append(w)
                
            k = m*(w**2)
            c = 2*damping*(k*m)**0.5
            
            u_e = sde[pp]
            u_y = u_e/Ry
            f_y = k*u_y
            
            
            
            # - Initial conditions for Central differences
            u = []
            u0 = 0.0
            a0 = -1*acc[0]
            u1 = u0 + 0.5*a0*dt**2
            u.append(u0)
            u.append(u1)
            F = []
            F0 = 0.0
            F1 = u1*k
            F.append(F0)
            F.append(F1)

            # - Run central differences
            for i in range(1,n-1):
                u_next = self.disp_next(u[i],u[i-1],acc[i],k,c,m,dt,F_i = F[i])
                F_next = self.force_next(u_next,u[i],u[i-1],k,F[i],f_y,alpha=alpha)
                u.append(u_next)
                F.append(F_next)

            sd = max(max(u),abs(min(u)))
            SD_nl.append(sd)
            
            # - Ductility computation 
            ductility = sd/u_y
            nu.append(ductility)
        
        SD_nl = np.array(SD_nl)
        omega = np.array(omega)
        SV_nl = omega*SD_nl
        SA_nl = SD_nl*omega**2
        return SA_nl,SV_nl,SD_nl,nu

    ################################################################################
    def inelastic_response(self,period,acc,dt,damping,Ry,alpha=None):
        SA,SV,SD,F_elastic = self.elastic_spectrum([period],acc,dt,damping)
        
        SD_nl,SV_nl,SA_nl,nu,omega = [],[],[],[],[]
        m = 1.0
        n = len(acc)
        p = period
        
        u_e = SD[0]
        f_e = F_elastic[0]
        u_y = u_e/Ry
        f_y = f_e/Ry

        w = 2*pi/p
        omega.append(w)
            
        k = m*(w**2)
        c = 2*damping*(k*m)**0.5
        
        # - Initial conditions for Central differences
        u = []
        u0 = 0.0
        a0 = -1*acc[0]
        u1 = u0 + 0.5*a0*dt**2
        u.append(u0)
        u.append(u1)
        F = []
        F0 = 0.0
        F1 = u1*k
        F.append(F0)
        F.append(F1)

        # - Run central differences
        for i in range(1,n-1):
            u_next = self.disp_next(u[i],u[i-1],acc[i],k,c,m,dt,F_i = F[i])
            F_next = self.force_next(u_next,u[i],u[i-1],k,F[i],f_y,alpha=alpha)
            u.append(u_next)
            F.append(F_next)

        sd = max(max(u),abs(min(u)))
        # - Ductility computation 
        ductility = sd/u_y

        return u,F,ductility

   ################################################################################     
    def disp_next(self,u_i,u_prev,x,k,c,m,dt,F_i=None):
        """
        Computes the next displacement point, based on central differences method.
        - Parameters:
            u_next: Next displacement
            u_i: Current displacement
            u_prev: previous displacement
            x: Accelation at moment 'i'
            k: stiffness
            c: damping coefficient
            m: mass
        """
        if F_i is None:
            u_next = 2*u_i - u_prev + ((-m*x -c*(u_i-u_prev)/dt - k*u_i)/(m + c*dt/2.0))*dt**2
        else:
            u_next = 2*u_i - u_prev + ((-m*x -c*(u_i-u_prev)/dt - F_i)/(m + c*dt/2.0))*dt**2
        return u_next
    ################################################################################   
    def force_next(self,u_next,u_i,u_prev,k,F_i,f_y,alpha=None):
        """
        Computes the next force point, based on assumed elastoplastic histeresis loop.
        - Parameters:
            F_next: Next force
            u_next: Next displacement
            u_i: Current displacement
            u_prev: previous displacement
            F_i: Current force 
            k: stiffness
        """
        du = u_next-u_i
        F_elastic = du*k + F_i
        
        if du > 0.0:
            fy = f_y
        else:
            fy = -f_y

        def elastic_branch():
            F = F_elastic
            return F
        
        def elastic_to_hardening_branch():
            u_ty = (fy-F_i)/k
            F = fy + (u_next -(u_i + u_ty))*alpha*k
            return F

        def hardening_branch():
            F = du*alpha*k + F_i
            return F
        
        # - Positive boundary case
        if fy > 0.0:
            if F_i <= fy:
                if F_elastic <= fy:
                    F_next = F_elastic
                else:
                    F_next = elastic_to_hardening_branch()
            else:
                F_next = hardening_branch()

        # - Negative boundary case
        else:
            if F_i >= fy:
                if F_elastic >= fy:
                    F_next = F_elastic
                else:
                    F_next = elastic_to_hardening_branch()
            else:
                F_next = hardening_branch()

        return F_next
    ################################################################################   
    def disp_origin(self,u_next,u_i,u_prev,u_o):
        """
        Computes the location of the displacement within the histeresis loop
        - Parameters:
            u_next: Next displacement
            u_i: Current displacement
            u_o: previous origin displacement
        """
        if (u_next-u_i)*(u_i-u_prev) < 0.0:
            u_oo = u_i
        else:
            u_oo = u_o
        return u_oo
    ################################################################################   
    def force_origin(self,u_next,u_i,u_prev,F_o,F_i):
        """
        Computes the location of the displacement within the histeresis loop
        - Parameters:
            u_next: Next displacement
            u_i: Current displacement
            F_o: previous origin force
            F_i: Current force at moment 'i'
        """
        if (u_next-u_i)*(u_i-u_prev) < 0.0:
            f_oo = F_i
        else:
            f_oo = F_o
        return f_oo
