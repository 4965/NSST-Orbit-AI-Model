import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')
def plot(rs):
    fig=plt.figure(figsize=(18,6))
    ax=fig.add_subplot(111,projection='3d')
    
    

    # Extract the x and y coordinates from the rs array
    x = rs[0, 0]
    y = rs[0, 2]

    # Plot the initial position
    ax.plot([x], [y], 'wo', label='initialposition')


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
    plt.show()
    

earth_radius = 6378.0       # km
earth_mu = 398600.0

def diffy_q(t, y, mu):
    # unpack state 
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx, ry, rz])

    # norm of the radius vector
    norm_r = np.linalg.norm(r)

    # two body acceleration
    ax, ay, az = -r * mu / norm_r**3

    return [vx, vy, vz, ax, ay, az]

# initial conditions of orbit parameters
r_mag = earth_radius + 500.0         # km
v_mag = np.sqrt(earth_mu / r_mag)    # km

# initial position and velocity vectors
r0 = [r_mag, 0, 0]
v0 = [0, v_mag, 0]

# time span
tspan = 100 * 60.0

# time step
dt = 100.0

# total number of steps
n_steps = int(np.ceil(tspan / dt))

# initialize arrays
ys = np.zeros((n_steps, 6))
ts = np.zeros((n_steps, 1))

# initial conditions
y0 = r0 + v0
ys[0] = np.array(y0)
step = 1

# initial solver
solver = ode(diffy_q)
solver.set_integrator('lsoda')
solver.set_initial_value(y0, 0)
solver.set_f_params(earth_mu)

# propagate orbit
while solver.successful() and step < n_steps:
    solver.integrate(solver.t + dt)
    ts[step] = solver.t
    ys[step] = solver.y
    step += 1

rs = ys[:, :3]

plot(rs)





