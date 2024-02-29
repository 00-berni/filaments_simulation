import numpy as np
from numpy import pi
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import filpy.alglin as alg


t = np.linspace(0,10,500)
line_dir = lambda t :  np.array([t*(t-5) + (t-7)**2,
                                 t**2*(t-5),
                                 2*(t-5)**4-t**3])
# line axis
l = line_dir(t)
xl,yl,zl = l

# objects
r = 10

obj = lambda t, th0: np.array([r*np.cos(t*pi/5 + th0*pi) + xl,
                               4*r*np.sin(t*pi/5 + th0*pi) + yl,
                               zl])

part = np.array([obj(t,theta0) for theta0 in np.linspace(0,2,100)])
print(part[:,1].shape)
# angles
alpha = np.arctan2(np.sqrt(part[:,0]**2 + part[:,1]**2),part[:,2])
beta = np.arctan2(part[:,1],part[:,0])

# velocities
V = 2
v = V * np.array([np.cos(beta)*np.sin(alpha) + 3/V, np.sin(beta)*np.sin(alpha) + 3/V, np.cos(alpha)])

# line of sight
line_sight = np.array([t+5,
                       -t,
                       t])


map_xy = np.array([[-5,15,15,-5,-5],[-50,-50,50,50,-50]])
ind_z = np.where(np.logical_and(np.abs(part[:,1,:]) < -map_xy[1,-1], np.abs(part[:,0,:]-5) < 10))

map_yz = np.array([[-60,60,60,-60,-60],[-150,-150,150,150,-150]])
ind_x = np.where(np.logical_and(np.abs(part[:,2,:]) < -map_yz[1,-1], np.abs(part[:,1,:]) < -map_yz[0,-1]))

map_zx = np.array([[-200,-200,0,0,-200],[-20,20,20,-20,-20]])
ind_y = np.where(np.logical_and(np.abs(part[:,0,:]) < -map_yz[1,-1], np.abs(part[:,2,:]-100) < 100))



ax = plt.figure().add_subplot(projection='3d')
ax.plot(*l,linestyle='dashdot',color='orange')
for p in part:
    ax.plot(*p,color='blue',alpha=0.1)
# ax.plot(*line_sight,linestyle='dashed',color='green')

plt.figure()
plt.subplot(231)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(xl,yl,'-.',color='orange')
plt.plot(*map_xy,color='red')
for p in part:
    plt.plot(p[0],p[1],color='blue',alpha=0.1)
plt.subplot(232)
plt.ylabel('z')
plt.xlabel('y')
plt.plot(yl,zl,'-.',color='orange')
plt.plot(*map_yz,color='red')
for p in part:
    plt.plot(p[1],p[2],color='blue',alpha=0.1)
plt.subplot(233)
plt.xlabel('z')
plt.ylabel('x')
plt.plot(zl,xl,'-.',color='orange')
plt.plot(*map_zx,color='red')
for p in part:
    plt.plot(p[2],p[0],color='blue',alpha=0.1)

plt.subplot(234)
plt.hist(v[2,ind_z[0],ind_z[1]].flatten(),100)
plt.xlabel('$v_z$')
plt.ylabel('counts')
plt.subplot(235)
plt.hist(v[0,ind_x[0],ind_x[1]].flatten(),100)
plt.xlabel('$v_x$')
plt.subplot(236)
plt.hist(v[1,ind_y[0],ind_y[1]].flatten(),100)
plt.xlabel('$v_y$')

print(v.shape)


plt.show()