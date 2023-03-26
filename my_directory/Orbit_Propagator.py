import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
import planatary_data as pd

class orbitpropagator:
    def _init_(self,r0,v0,tspan,dt,cb=pd.earth):
        self.r0=r0
        self.v0=v0
        self.y0=self.r0.tolist()+self.v0.tolist()
        self.tspan=tspan
        self.dt=dt
        self.cb=cb

    def prograte_orbit(self):

        # total number of steps
        self.n_steps = int(np.ceil(self.tspan /self. dt))

        # initialize arrays
        self.ys = np.zeros((self.n_steps, 6))
        self.ts = np.zeros((self.n_steps, 1))

        # initial variables
        self.ts =np.zeros((self.n_steps,1))
        self.ys =np.zeros((self.n_steps,6))
        self.y0=self.r0+self.v0       #initial conditions
        self.ts[0]=0
        self.ys[0]=self.y0
        self.step=1

        # initial solver
        self.solver = ode(self.diffy_q)
        self.solver.set_integrator('lsoda')
        self.solver.set_initial_value(self.y0, 0)


        # propagate orbit
        while self.solver.successful() and self.step <self.n_steps:
             self.solver.integrate(self.solver.t + self.dt)
             self.ts[self.step] = self.solver.t
             self.ys[self.step] = self.solver.y
             self.step += 1

        self.rs = self.ys[:, :3]
        self.vs = self.ys[:, 3:]
        
    def diffy_q(self,t, y):
        # unpack state 
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])
        v = np.array ([rx, ry, rz])
        
        # norm of the radius vector
        norm_r = np.linalg.norm(r)

        # two body acceleration
        ax, ay, az = -r*self.cb['mu'] / norm_r**3

        return [vx, vy, vz, ax, ay, az]
        
        
    def plot_3d(self,show_plot=false,save_plot=false):
        fig=plt.figure(figsize=(16,8))
        ax=fig.add_subplot(111,projection='3d')
    
        
        #plot trajectory
        ax.plot(self.rs[:,0],self.rs[:,1],self.rs[:,2],'w',label='Trajectory')
        ax.plot([self.rs[0,0]],[self.rs[0,1]],[self.rs[0,2]],'wo',label='initial position')

        
        


        #plot central body
        _u,_v=np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
        _x=earth_radius*np.cos(_u)*np.sin(_v)
        _y=earth_radius*np.sin(_u)*np.sin(_v)
        _z=earth_radius*np.cos(_v)*np.sin(_v) 
        ax.plot_surface(_x,_y,_z,cmap='Blues')
    
        #plot the x,y,z vectors
        1==earth_radius*2
        x,y,z=[[0,0,0],[0,0,0],[0,0,0]]
        u,v,w=[[1,0,0],[0,1,0],[0,0,1]]
        ax.quiver(x,y,z,u,v,w,color='k')
    
        max_val=np.max(np.abs(rs))
        ax.set_xlim([-max_val,max_val])
        ax.set_ylim([-max_val,max_val])
        ax.set_zlim([-max_val,max_val])
    
        ax.set_xlabel(['x(Km)'])
        ax.set_ylabel(['y(Km)'])
        ax.set_zlabel(['z(Km)'])
    
        #ax.set_aspect('equal')
        ax.set_title('Example title')
        plt.legend()
        
        if show_plot:
            plt.show()
        if save_plot:
            plt.savefig(title+'.png',dpi=300)
            
        
        
        
        
        
        
        
        
        
        
     
