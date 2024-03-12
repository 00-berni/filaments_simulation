from typing import Any
import numpy as np
from numpy import pi, cos, sin
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import filpy.alglin as alg


test_num = 0

if test_num == 0:

    DISPLAY_PLOT = True     #: parameter to trigger the plot representation

    ## Initialization
    numpoints = 200         #: number of points
    u = np.linspace(-0.2,1.2,numpoints)     #: parametrization
    
    # compute the curve
    x_i = (u+1)**2
    x_j = sin(u*pi)
    x_k = x_i * x_j
    x = np.array([x_i,x_j,x_k])     #: curve vector

    # compute the tangent vector to the curve
    t_i = 2*(u+1)
    t_j = cos(u*pi)*pi
    t_k = t_i * x_j + x_i * t_j
    # normalize it
    t = np.array([t_i,t_j,t_k])/np.sqrt(t_i**2+t_j**2+t_k**2) 

    # set the initial values for the other orthogonal vectors
    r = np.array([[1],[0],[-t_i[0]/t_k[0]]])    #: normal vector
    # normalize it
    r /= np.sqrt(np.sum(r**2,axis=0))

    s = np.cross(t[:,0],r[:,0]).reshape(3,1)    #: third vector
    # normalize it
    s /= np.sqrt(np.sum(s**2,axis=0))

    # compute the frame along the curve via the double reflection method
    r,s,t = alg.double_reflection(x,t,r,s,numpoints)

    ## Computation of the streamlines
    R = 0.2     #: cylinder radius
    
    def compute_stream(R: float = R,th0: float = 0) -> NDArray:
        """Function to compute a streamline along the curve
        """
        # compute the streamline in the frame of the curve
        stream = np.array([R*cos(u*pi + th0*pi),R*sin(u*pi + th0*pi),np.zeros(numpoints)])
        # compute the matrix to change frame reference
        mat = [np.array([r[:,ui],s[:,ui],t[:,ui]]).T for ui in range(numpoints) ]
        # compute the cartesian coordinates of the streamline
        return np.array([ np.dot(mat[ui],stream[:,ui]) for ui in range(numpoints)]).T + x

    stream1 = compute_stream()
    stream2 = compute_stream(th0=1.5)

    ## Plotting    
    if DISPLAY_PLOT:
        PLOT_PARAM = 0

        if PLOT_PARAM == 0:
            import mayavi.mlab as mlab
            # pl1 = mlab.plot3d(*x,color=(0,1,0),name='cylinder axis',tube_radius=R)
            pl2 = mlab.plot3d(*x,color=(0,1,0),name='cylinder axis',tube_radius=None)

            th0_max = 1.9
            num = 20
            for th0 in np.linspace(0,th0_max,num):
                mlab.plot3d(*compute_stream(R,th0),color=(0,0,1))
            # mlab.plot3d(*stream1,color=(1,0,0))
            # mlab.plot3d(*stream2,color=(0,0,1))
            mlab.show()
        elif PLOT_PARAM == 1:
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot(*x,'-.',color='orange')
            num = 3
            for th0 in np.linspace(0,1.8,num):
                ax.plot(*compute_stream(R,th0),color='blue')
            # for ui in range(numpoints):
            #     ax.plot(*np.append(x[:,ui].reshape(3,1),(r+x)[:,ui].reshape(3,1),axis=1),color='red')
            #     ax.plot(*np.append(x[:,ui].reshape(3,1),(s+x)[:,ui].reshape(3,1),axis=1),color='green')
            # ax.plot(*stream2)

            plt.show()