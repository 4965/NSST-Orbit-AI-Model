import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3
plt.style.use('dark background')
from sys import path

path.append('/NSST-Orbit-AI-Model/my_directory/python_tools')\
from Orbit_Propagator import orbitpropagator as op
import planatery_data as pd

op=op(r0,v0,tspan,dt,cb=cb)
op.propagate_orbit()
op.plot_3d(show_plot=True)


cb=pd.sun

if __name__ == '__main__':
    #initial magnitude of orbit
    r_mag = cb['radius'] + cb['radius']*4       
    v_mag = np.sqrt(cb['mu'] / r_mag)
    
    
    #initial position and velosity vectors
    r0 = [r_mag,r_mag*0.01,r_mag*-0.5]
    v0 = [0,v_mag,v_mag*0.8]
    
    #1 day
    tspan = 3600*24.0
    
    #100 seconds
    dt = 100.0
