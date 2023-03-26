import numpy as np
import matplotlib.pyplot as plt
import planatary_data as pd

d2r=np.pi/180.0

def plot_n_orbits(rs,labels,cb=pd.earth,show_plot=False,save_plot=False,title='Many orbits'):
        fig=plt.figure(figsize=(16,8))
        ax=fig.add_subplot(111,projection='3d')
    
        
        #plot trajectory
        n=0
        for r in rs:
           ax.plot(r[:,0],r[:,1],r[:,2],label=labels[n])
           ax.plot([r[0,0]],[r[0,1]],[r[0,2]])
           n += 1

        
        


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
            
