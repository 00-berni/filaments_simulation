import numpy as np
from numpy import pi
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import filpy.alglin as alg

numpoints = 500                     #: number of points
t = np.linspace(0,10,numpoints)     #: parameter for the curve (like time istants)

# define axis curve function
line_dir = lambda t :  np.array([t*(t-5) + (t-7)**2,    #: x
                                 t**2*(t-5),            #: y
                                 2*(t-5)**4-t**3])      #: z
l = line_dir(t)     #: curve axis
xl,yl,zl = l        #: its coordinates

r = 10              #: cylinder radius

# define the equation for the stream trajectories
# depending on the parameter `t` and the initial angle `th0`
obj = lambda t, th0: np.array([r*np.cos(t*pi/5 + th0*pi) + xl,
                               4*r*np.sin(t*pi/5 + th0*pi) + yl,
                               zl])

N = 100     #: number of streams
# collect the stream, computing their trajectories
part = np.array([obj(t,theta0) for theta0 in np.linspace(0,2,100)])

# compute velocities
# projection angles
alpha = np.arctan2(np.sqrt(part[:,0]**2 + part[:,1]**2),part[:,2])
beta = np.arctan2(part[:,1],part[:,0])

V = 2       #: modulus of the velocity
# compute velocity components for each stream
v = V * np.array([[np.cos(b)*np.sin(a), np.sin(b)*np.sin(a), np.cos(a)] for (a,b) in zip(alpha,beta)])


# define selected areas to study velocity
map_xy = np.array([[-5,15,15,-5,-5],[-50,-50,50,50,-50]])
ind_z = np.where(np.logical_and(np.abs(part[:,1,:]) < -map_xy[1,-1], np.abs(part[:,0,:]-5) < 10))

map_yz = np.array([[-60,60,60,-60,-60],[-150,-150,150,150,-150]])
ind_x = np.where(np.logical_and(np.abs(part[:,2,:]) < -map_yz[1,-1], np.abs(part[:,1,:]) < -map_yz[0,-1]))

map_zx = np.array([[-200,-200,0,0,-200],[-20,20,20,-20,-20]])
ind_y = np.where(np.logical_and(np.abs(part[:,0,:]) < -map_yz[1,-1], np.abs(part[:,2,:]-100) < 100))


## 3D plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(*l,linestyle='dashdot',color='orange')
for (pi,vi) in zip(part,v):
    ax.plot(*pi,color='blue',alpha=0.1)

## projections
fig, axs = plt.subplots(2,3)
pxy = axs[:,0]
pyz = axs[:,1]
pzx = axs[:,2]
# normalize colormap
vmin, vmax = (v).min(), (v).max()

# xy plane
pxy[0].set_xlabel('x')
pxy[0].set_ylabel('y')
pxy[0].plot(*map_xy,color='yellow')
for (pi,vi) in zip(part,v):
    pxy[0].scatter(pi[0],pi[1],c=vi[2],vmin=vmin,vmax=vmax,cmap='RdBu')
pxy[1].hist(v[ind_z[0],2,ind_z[1]].flatten(),100)
pxy[1].set_xlabel('$v_z$')
pxy[1].set_ylabel('counts')

# yz plane
pyz[0].set_ylabel('z')
pyz[0].set_xlabel('y')
pyz[0].plot(*map_yz,color='yellow')
for (pi,vi) in zip(part,v):
    pyz[0].scatter(pi[1],pi[2],c=vi[0],vmin=vmin,vmax=vmax,cmap='RdBu')
pyz[1].hist(v[ind_x[0],0,ind_x[1]].flatten(),100)
pyz[1].set_xlabel('$v_x$')

# zx plane
pzx[0].set_xlabel('z')
pzx[0].set_ylabel('x')
pzx[0].plot(*map_zx,color='yellow')
for (pi,vi) in zip(part,v):
    pp = pzx[0].scatter(pi[2],pi[0],c=vi[1],vmin=vmin,vmax=vmax,cmap='RdBu')
cbaxes = fig.add_axes([0.92, 0.1, 0.02, 0.8])  
fig.colorbar(pp,cax = cbaxes,extend='both',label='$v_i$')
pzx[1].hist(v[ind_y[0],1,ind_y[1]].flatten(),100)  
pzx[1].set_xlabel('$v_y$')



plt.show()