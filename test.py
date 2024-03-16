import numpy as np
from numpy import pi, cos, sin
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import filpy.geobj as geobj
from filpy.geobj import Frame, Filament

test_num = 1

if test_num == 0:


    ## Initialization
    numpoints = 500                          #: number of points
    u = np.linspace(-2.3,-0.4,numpoints)     #: parametrization
    
    a1 = 1/2
    w = 2
    # compute the curve
    x_i = u**2/a1
    x_j = sin(w*u)**2
    x_k = x_i * x_j / 2 
    x = np.array([x_i,x_j,x_k])     #: curve vector


    # compute the tangent vector to the curve
    t_i = 2*u/a1
    t_j = 2*w*cos(w*u)*sin(w*u)
    t_k = (t_i*x_j + t_j*x_i) / 2
    # normalize it
    t = Frame.normalize(np.array([t_i,t_j,t_k])) 

    # set the initial values for the other orthogonal vectors
    r = Frame.normalize(np.array([[1],[0],[-t[0,0]/t[2,0]]]))    #: normal vector

    s = Frame.normalize(np.cross(t[:,0],r[:,0]).reshape(3,1))    #: third vector

    # compute the frame evolution along the curve via the double reflection method
    frame = Frame(u,x,t,r,s)

    # rot
    rot = True
    if rot:
        ang_i = (1+13/180)
        ang_j = -(15/180)
        ang_k = (1+0.5/180)

        frame.rotate((ang_i, ang_j, ang_k), 'ijk')


    ## Computation of the filament line
    omega = 0.1       #: wrapping
    R_max = 0.2     #: cylinder radius
    R_min = 0.1
    r_den = 5
    R = np.linspace(R_min,R_max,r_den)
    th_max = 1.98
    th_den = 30
    th0 = np.linspace(0,th_max,th_den)
    V = lambda v1 : v1
    v_field = 's'
    V = lambda v1,v2 : v1-v2*omega
    v_field = 'rs'
    # V = 5
    # v_field = 'unif'
    filament = Filament(frame,(omega,R,th0),V,field=v_field)
    trajectory = filament.trj
    velocity   = filament.vel

    ## Set maps

    i = (1.75,3.00)
    j = (0.80,1.45)
    k = (7.10,8.00)

    cond_str_i = (trajectory[:,0] >= i[0]) & (trajectory[:,0] <= i[1])
    cond_str_j = (trajectory[:,1] >= j[0]) & (trajectory[:,1] <= j[1])
    cond_str_k = (trajectory[:,2] >= k[0]) & (trajectory[:,2] <= k[1])

    # # indx_i = np.where(cond_str_j & cond_str_k)
    # # indx_j = np.where(cond_str_k & cond_str_i)
    # # indx_k = np.where(cond_str_i & cond_str_j)
    # indx_i = None
    # indx_j = None
    # indx_k = None

    print(np.mean(velocity[:,0]))
    print(np.mean(velocity[:,1]))
    print(np.mean(velocity[:,2]))
     

    # ## Plotting    
    DISPLAY_PLOT = True     #: parameter to trigger the plot representation

    if DISPLAY_PLOT:
            
        plots = '0'
#         plots = input(#
# """
# Select a plot to show:
#     (0) 3D rapresentation
#     (1) 3 projections
#     (2) i-projection
#     (3) i-projection without bulk motion
# If you want a combination of them simply write the numbers you choose
# If you want all use 'all'\n
# """
#                     )

        # map1 = ((-0.25,0.2),(0.9,1.6))
        # map1 = ((-0.09,0.09),(1.2,1.6))
        map1 = ((-0.1,0.2),(0.4,0.8))
        map2 = ((0.3,0.8),(-0.6,-0.))
        map3 = ((-0.5,-0.2),(-2.35,-2.05))

        if plots == 'all' or '0' in plots:
            import mayavi.mlab as mlab
            # pl1 = mlab.plot3d(*x,color=(0,1,0),name='cylinder axis',tube_radius=R)
            pl2 = mlab.plot3d(*frame.line,color=(0,1,0),name='cylinder axis',tube_radius=None)
            colors = np.array([np.linspace(0,1,len(trajectory)//r_den)]*r_den).flatten()
            for (trj,c1) in zip(trajectory,colors):
                mlab.plot3d(*trj,color=(c1,0,1),tube_radius=None)
            
            mlab.show()

        vmin, vmax = velocity.min(), velocity.max()
        binning = 100
        print(vmin,vmax)
        if plots == 'all' or '1' in plots:
            fig, ax = plt.subplots(2,3)

            fig.suptitle(f'{len(filament.lines)} trajectory with {numpoints} points\n$\ell = i, j, k$, bins num: {binning}, $R \\in [{R_min}, {R_max}]$, $\\omega =$ {omega}, '+'uniform velocity: $|\\vec{V}| =$ '+f'{V}')

            axi = ax[:,0]
            axj = ax[:,1]
            axk = ax[:,2]
            
            axi[0].set_title('Plane $j$-$k$')
            for ui in range(numpoints):
                axi[0].scatter(trajectory[:,1,ui],trajectory[:,2,ui],c=velocity[:,0,ui],vmin=vmin,vmax=vmax,cmap='RdBu')
            # axi[0].plot([j[0],j[0],j[1],j[1],j[0]],[k[0],k[1],k[1],k[0],k[0]],color='green')
            axi[0].set_xlabel('$j$')
            axi[0].set_ylabel('$k$')
            axi[1].hist(velocity[:,0,:].flatten(),bins=binning)
            axi[1].set_xlabel('$v_i$')
            axi[1].set_ylabel('$counts$')
        
            axj[0].set_title('Plane $k$-$i$')
            for ui in range(numpoints):
                axj[0].scatter(trajectory[:,2,ui],trajectory[:,0,ui],c=velocity[:,1,ui],vmin=vmin,vmax=vmax,cmap='RdBu')
            # axj[0].plot([k[0],k[0],k[1],k[1],k[0]],[i[0],i[1],i[1],i[0],i[0]],color='green')
            axj[0].set_xlabel('$k$')
            axj[0].set_ylabel('$i$')
            axj[1].hist(velocity[:,1,:].flatten(),bins=binning)
            axj[1].set_xlabel('$v_j$')
            
            axk[0].set_title('Plane $i$-$j$')
            for ui in range(numpoints):
                plot_k = axk[0].scatter(trajectory[:,0,ui],trajectory[:,1,ui],c=velocity[:,2,ui],vmin=vmin,vmax=vmax,cmap='RdBu')
            # axk[0].plot([i[0],i[0],i[1],i[1],i[0]],[j[0],j[1],j[1],j[0],j[0]],color='green')
            axk[0].set_xlabel('$i$')
            axk[0].set_ylabel('$j$')
            axk[1].hist(velocity[:,2,:].flatten(),bins=binning)
            axk[1].set_xlabel('$v_k$')

            cbaxes = fig.add_axes([0.92, 0.1, 0.02, 0.8])  
            fig.colorbar(plot_k,cax = cbaxes,extend='both',label='$v_\\ell$')
            
            plt.show()

        if plots == 'all' or '2' in plots:
            fig = plt.figure()
            fig.suptitle(f'Plane $j$-$k$ - $\\omega =$ {omega}')
            ax0 = fig.add_subplot(1,2,1)
            for ui in range(numpoints):
                pp = ax0.scatter(trajectory[:,1,ui],trajectory[:,2,ui],c=velocity[:,0,ui],vmin=vmin,vmax=vmax,cmap='RdBu')
            ax0.set_xlabel('$j$')
            ax0.set_ylabel('$k$')
            
            
            ax0.plot([map1[0][0],map1[0][0],map1[0][1],map1[0][1],map1[0][0]],[map1[1][0],map1[1][1],map1[1][1],map1[1][0],map1[1][0]], color='green',label='map1')
            ax0.plot([map2[0][0],map2[0][0],map2[0][1],map2[0][1],map2[0][0]],[map2[1][0],map2[1][1],map2[1][1],map2[1][0],map2[1][0]], color='violet',label='map2')
            ax0.plot([map3[0][0],map3[0][0],map3[0][1],map3[0][1],map3[0][0]],[map3[1][0],map3[1][1],map3[1][1],map3[1][0],map3[1][0]], color='orange',label='map3')

            ax0.legend()

            cbaxes = fig.add_axes([0.02, 0.1, 0.02, 0.8])  
            fig.colorbar(pp,cax = cbaxes,extend='both',label='$v_i$')
            
            indx1 = np.where((trajectory[:,1] >= map1[0][0]) & (trajectory[:,1] <= map1[0][1]) & (trajectory[:,2] >= map1[1][0]) & (trajectory[:,2] <= map1[1][1]))
            indx2 = np.where((trajectory[:,1] >= map2[0][0]) & (trajectory[:,1] <= map2[0][1]) & (trajectory[:,2] >= map2[1][0]) & (trajectory[:,2] <= map2[1][1]))
            indx3 = np.where((trajectory[:,1] >= map3[0][0]) & (trajectory[:,1] <= map3[0][1]) & (trajectory[:,2] >= map3[1][0]) & (trajectory[:,2] <= map3[1][1]))

            ax1 = fig.add_subplot(3,2,2)
            ax1.hist(velocity[:,0,indx1].flatten(),bins=binning,color='green',label='map1')
            ax1.set_ylabel('counts')
            ax1.legend()
            ax2 = fig.add_subplot(3,2,4)
            ax2.hist(velocity[:,0,indx2].flatten(),bins=binning,color='violet',label='map2')
            ax2.set_ylabel('counts')
            ax2.legend()
            ax3 = fig.add_subplot(3,2,6)
            ax3.hist(velocity[:,0,indx3].flatten(),bins=binning,color='orange',label='map3')
            ax3.set_xlabel('$v_i$')
            ax3.set_ylabel('counts')
            ax3.legend()

            plt.show()

        if plots == 'all' or '3' in plots:
            v_bulk = np.mean(velocity[:,0])
            vel_i = velocity[:,0] - v_bulk
            vmin_i, vmax_i = vel_i.min(), vel_i.max()
            fig = plt.figure()
            fig.suptitle(f'Plane $j$-$k$ - $\\omega =$ {omega},'+ ' $v_{i,bulk} = $' + f'{v_bulk:.3}')
            ax0 = fig.add_subplot(1,2,1)
            for ui in range(numpoints):
                pp = ax0.scatter(trajectory[:,1,ui],trajectory[:,2,ui],c=vel_i[:,ui],vmin=vmin_i,vmax=vmax_i,cmap='RdBu')
            ax0.set_xlabel('$j$')
            ax0.set_ylabel('$k$')
                        
            ax0.plot([map1[0][0],map1[0][0],map1[0][1],map1[0][1],map1[0][0]],[map1[1][0],map1[1][1],map1[1][1],map1[1][0],map1[1][0]], color='green',label='map1')
            ax0.plot([map2[0][0],map2[0][0],map2[0][1],map2[0][1],map2[0][0]],[map2[1][0],map2[1][1],map2[1][1],map2[1][0],map2[1][0]], color='violet',label='map2')
            ax0.plot([map3[0][0],map3[0][0],map3[0][1],map3[0][1],map3[0][0]],[map3[1][0],map3[1][1],map3[1][1],map3[1][0],map3[1][0]], color='orange',label='map3')

            ax0.legend()

            cbaxes = fig.add_axes([0.02, 0.1, 0.02, 0.8])  
            fig.colorbar(pp,cax = cbaxes,extend='both',label='$v_i$')
            
            indx1 = np.where((trajectory[:,1] >= map1[0][0]) & (trajectory[:,1] <= map1[0][1]) & (trajectory[:,2] >= map1[1][0]) & (trajectory[:,2] <= map1[1][1]))
            indx2 = np.where((trajectory[:,1] >= map2[0][0]) & (trajectory[:,1] <= map2[0][1]) & (trajectory[:,2] >= map2[1][0]) & (trajectory[:,2] <= map2[1][1]))
            indx3 = np.where((trajectory[:,1] >= map3[0][0]) & (trajectory[:,1] <= map3[0][1]) & (trajectory[:,2] >= map3[1][0]) & (trajectory[:,2] <= map3[1][1]))

            ax1 = fig.add_subplot(3,2,2)
            ax1.hist(vel_i[:,indx1].flatten(),bins=binning,color='green',label='map1')
            ax1.set_ylabel('counts')
            ax1.legend()
            ax2 = fig.add_subplot(3,2,4)
            ax2.hist(vel_i[:,indx2].flatten(),bins=binning,color='violet',label='map2')
            ax2.set_ylabel('counts')
            ax2.legend()
            ax3 = fig.add_subplot(3,2,6)
            ax3.hist(vel_i[:,indx3].flatten(),bins=binning,color='orange',label='map3')
            ax3.set_xlabel('$v_i$')
            ax3.set_ylabel('counts')
            ax3.legend()

            plt.show()

elif test_num == 1:

    ## INITIALIZATION

    numpoints = 500
    u = np.linspace(-2,0, numpoints)

    # spine curve coordinates
    x_i = u**2
    x_j = (u+1)**2
    x_k = (u+1)**3
    x = np.array([x_i,x_j,x_k])

    # tangent vector coordinates
    t_i = 2*u
    t_j = 2*(u+1)
    t_k = 3*(u+1)**2
    t = Frame.normalize(np.array([t_i,t_j,t_k]))

    # start frame vectors
    r = Frame.normalize(np.array([[1],[1],[(-t[0,0]-t[0,1])/t[0,2]]]))
    s = Frame.normalize(np.cross(t[:,0],r[:,0]).reshape(3,1))

    # compute the frame
    frame = Frame(u,x,t,r,s)

    ## FILAMENT
    omega = 0
    r_den = 2
    R = np.linspace(0.2,0.3,r_den)
    th_den = 10
    th0 = np.linspace(0,1.9,th_den)
    V = 2
    filament1 = Filament(frame,(omega,R,th0),V,'unif')
    trajectory1 = filament1.trj

    ## PLOTS 
    DISPLAY_PLOT = True

    if DISPLAY_PLOT:

        plots = '0'

        if plots == 'all' or '0' in plots:
            import mayavi.mlab as mlab

            mlab.plot3d(*frame.line,color=(0,1,0),tube_radius=None)
            mlab.plot3d(*frame.line,color=(0,1,0),tube_radius=R[0])
            colors = np.array([np.linspace(0,1,len(trajectory1)//r_den)]*r_den).flatten()
            for (trj,c) in zip(trajectory1,colors):
                mlab.plot3d(*trj,color=(c,0,1))

            mlab.show()    