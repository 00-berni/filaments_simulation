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
    numpoints = 300                         #: number of points
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

    # rot
    ang = 15/180*pi
    rot_mat = np.array([[cos(ang),-sin(ang),0],
                        [sin(ang), cos(ang),0],
                        [0.      , 0.      ,1]])
    x = np.array([np.dot(rot_mat,x[:,ui]) for ui in range(numpoints)]).T
    x_i,x_j,x_k = x

    t = np.array([np.dot(rot_mat,t[:,ui]) for ui in range(numpoints)]).T
    t /= np.sqrt(np.sum(t**2,axis=0))
    t_i,t_j,t_k = t

    r = np.array([np.dot(rot_mat,r[:,ui]) for ui in range(numpoints)]).T
    r /= np.sqrt(np.sum(r**2,axis=0))

    s = np.array([np.dot(rot_mat,s[:,ui]) for ui in range(numpoints)]).T
    s /= np.sqrt(np.sum(s**2,axis=0))

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
    
    th_max = 1.98
    num = 40
    streamlines = np.array([compute_stream(R=R,th0=th0) for th0 in np.linspace(0,th_max,num)])

    ## Compute the velocity field
    # uniform field case
    V = 3
    v = np.array([V*t]*num)

    ## Set maps

    i = (1.75,3.00)
    j = (1.20,1.90)
    k = (1.80,2.80)

    cond_str_i = (streamlines[:,0] >= i[0]) & (streamlines[:,0] <= i[1])
    cond_str_j = (streamlines[:,1] >= j[0]) & (streamlines[:,1] <= j[1])
    cond_str_k = (streamlines[:,2] >= k[0]) & (streamlines[:,2] <= k[1])

    indx_i = np.where(cond_str_j & cond_str_k)
    indx_j = np.where(cond_str_k & cond_str_i)
    indx_k = np.where(cond_str_i & cond_str_j)
     

    ## Plotting    
    if DISPLAY_PLOT:
        PLOT_PARAM = 0

        if PLOT_PARAM == 0:
            
            plots = '2'
            if plots == 'all' or '0' in plots:
                import mayavi.mlab as mlab

                for stream in streamlines:
                    mlab.plot3d(*stream,color=(0,0,1),tube_radius=0.01)
                
                mlab.show()

            vmin, vmax = v.min(), v.max()
            binning = 50 # np.linspace(vmin,vmax,50)
            if plots == 'all' or '1' in plots:
                fig, ax = plt.subplots(2,3)

                fig.suptitle(f'{num} streamlines with {numpoints} points: projections $\ell = i, j, k$')

                axi = ax[:,0]
                axj = ax[:,1]
                axk = ax[:,2]
                
                axi[0].set_title('Plane $j$-$k$')
                for ui in range(numpoints):
                    axi[0].scatter(streamlines[:,1,ui],streamlines[:,2,ui],c=v[:,0,ui],vmin=vmin,vmax=vmax,cmap='RdBu')
                axi[0].plot([j[0],j[0],j[1],j[1],j[0]],[k[0],k[1],k[1],k[0],k[0]],color='green')
                axi[0].set_xlabel('$j$')
                axi[0].set_ylabel('$k$')
                axi[1].hist(v[:,0,indx_i].flatten(),bins=binning)
                axi[1].set_xlabel('$v_i$')
                axi[1].set_ylabel('$counts$')
            
                axj[0].set_title('Plane $k$-$i$')
                for ui in range(numpoints):
                    axj[0].scatter(streamlines[:,2,ui],streamlines[:,0,ui],c=v[:,1,ui],vmin=vmin,vmax=vmax,cmap='RdBu')
                axj[0].plot([k[0],k[0],k[1],k[1],k[0]],[i[0],i[1],i[1],i[0],i[0]],color='green')
                axj[0].set_xlabel('$k$')
                axj[0].set_ylabel('$i$')
                axj[1].hist(v[:,1,indx_j].flatten(),bins=binning)
                axj[1].set_xlabel('$v_j$')
                
                axk[0].set_title('Plane $i$-$j$')
                for ui in range(numpoints):
                    plot_k = axk[0].scatter(streamlines[:,0,ui],streamlines[:,1,ui],c=v[:,2,ui],vmin=vmin,vmax=vmax,cmap='RdBu')
                axk[0].plot([i[0],i[0],i[1],i[1],i[0]],[j[0],j[1],j[1],j[0],j[0]],color='green')
                axk[0].set_xlabel('$i$')
                axk[0].set_ylabel('$j$')
                axk[1].hist(v[:,2,indx_k].flatten(),bins=binning)
                axk[1].set_xlabel('$v_k$')

                cbaxes = fig.add_axes([0.92, 0.1, 0.02, 0.8])  
                fig.colorbar(plot_k,cax = cbaxes,extend='both',label='$v_\\ell$')
                
                plt.show()

            if plots == 'all' or '2' in plots:
                fig = plt.figure()
                fig.suptitle('Plane $j$-$k$')
                ax0 = fig.add_subplot(1,2,1)
                for ui in range(numpoints):
                    pp = ax0.scatter(streamlines[:,1,ui],streamlines[:,2,ui],c=v[:,0,ui],vmin=vmin,vmax=vmax,cmap='RdBu')
                ax0.set_xlabel('$j$')
                ax0.set_ylabel('$k$')
                
                # map1 = ((0.35,0.75),(0.9,1.55))
                map1 = ((1.25,1.60),(1.7,2.1))
                map2 = (j,k)
                map3 = ((0.63,0.93),(-2.2,-1.6))
                
                ax0.plot([map1[0][0],map1[0][0],map1[0][1],map1[0][1],map1[0][0]],[map1[1][0],map1[1][1],map1[1][1],map1[1][0],map1[1][0]], color='green',label='map1')
                ax0.plot([map2[0][0],map2[0][0],map2[0][1],map2[0][1],map2[0][0]],[map2[1][0],map2[1][1],map2[1][1],map2[1][0],map2[1][0]], color='violet',label='map2')
                ax0.plot([map3[0][0],map3[0][0],map3[0][1],map3[0][1],map3[0][0]],[map3[1][0],map3[1][1],map3[1][1],map3[1][0],map3[1][0]], color='orange',label='map3')

                ax0.legend()

                cbaxes = fig.add_axes([0.02, 0.1, 0.02, 0.8])  
                fig.colorbar(pp,cax = cbaxes,extend='both',label='$v_i$')
                
                indx1 = np.where((streamlines[:,1] >= map1[0][0]) & (streamlines[:,1] <= map1[0][1]) & (streamlines[:,2] >= map1[1][0]) & (streamlines[:,2] <= map1[1][1]))
                indx2 = np.where((streamlines[:,1] >= map2[0][0]) & (streamlines[:,1] <= map2[0][1]) & (streamlines[:,2] >= map2[1][0]) & (streamlines[:,2] <= map2[1][1]))
                indx3 = np.where((streamlines[:,1] >= map3[0][0]) & (streamlines[:,1] <= map3[0][1]) & (streamlines[:,2] >= map3[1][0]) & (streamlines[:,2] <= map3[1][1]))

                ax1 = fig.add_subplot(3,2,2)
                ax1.hist(v[:,0,indx1].flatten(),bins=binning,color='green',label='map1')
                ax1.set_ylabel('counts')
                ax1.legend()
                ax2 = fig.add_subplot(3,2,4)
                ax2.hist(v[:,0,indx2].flatten(),bins=binning,color='violet',label='map2')
                ax2.set_ylabel('counts')
                ax2.legend()
                ax3 = fig.add_subplot(3,2,6)
                ax3.hist(v[:,0,indx3].flatten(),bins=binning,color='orange',label='map3')
                ax3.set_xlabel('$v_i$')
                ax3.set_ylabel('counts')
                ax3.legend()

                plt.show()

        elif PLOT_PARAM == 1:
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot(*x,'-.',color='orange')
            th0_max = 1.9
            num = 10
            for th0 in np.linspace(0,th0_max,num):
                ax.plot(*compute_stream(R,th0),color='blue')



            plt.show()