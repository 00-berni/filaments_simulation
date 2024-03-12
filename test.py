import numpy as np
from numpy import pi, cos, sin
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import filpy.alglin as alg


test_num = 2

if test_num == 0:

    numpoints = 500                     #: number of points
    t = np.linspace(0,10,numpoints)     #: parameter for the curve (like time istants)

    # define axis curve function
    line_dir = lambda t :  np.array([t*(t-5) + (t-7)**2,    #: x
                                    (t+1)**2*(t-5),            #: y
                                    2*(t-5)**4-t**3 + 5])      #: z
    l = line_dir(t)     #: curve axis
    xl,yl,zl = l        #: its coordinates

    r = 15              #: cylinder radius

    # define the equation for the stream trajectories
    # depending on the parameter `t` and the initial angle `th0`
    obj = lambda t, th0: np.array([r*np.cos(t*pi/20 + th0*pi) + xl,
                                -10*r*np.sin(t*pi/20 + th0*pi) + yl,
                                zl])

    N = 10     #: number of streams
    # collect the stream, computing their trajectories
    part = np.array([obj(t,theta0) for theta0 in np.linspace(0,1,N)])
    print(f'part shape:\t{part.shape}')

    # compute velocities
    # projection angles
    alpha = np.arctan2(np.sqrt(part[:,0]**2 + part[:,1]**2),part[:,2])
    beta  = np.arctan2(part[:,1],part[:,0])

    V = 2       #: modulus of the velocity
    # compute velocity components for each stream
    v = V * np.array([[np.cos(b)*np.sin(a), np.sin(b)*np.sin(a), np.cos(a)] for (a,b) in zip(alpha,beta)])

    axis = 0
    gamma = 0
    if gamma != 0:
        mat = np.dot(alg.rotation(1,gamma),alg.rotation(axis,gamma))
        part = np.array([ np.dot(mat,p_i) for p_i in part])
        l = np.dot(mat,l)
        v = np.array([np.dot(mat,v_i) for v_i in v])

    # define selected areas to study velocity
    map_xy = np.array([[-10,25,25,-10,-10],[-60,-60,60,60,-60]])
    ind_z = np.where(np.logical_and(np.abs(part[:,1,:]) < -map_xy[1,-1], np.abs(part[:,0,:]-7.5) < 17.5))

    map_yz = np.array([[-60,60,60,-60,-60],[-150,-150,150,150,-150]])
    ind_x = np.where(np.logical_and(np.abs(part[:,2,:]) < -map_yz[1,-1], np.abs(part[:,1,:]) < -map_yz[0,-1]))

    map_zx = np.array([[-70,-70,70,70,-70],[-10,30,30,-10,-10]])
    ind_y = np.where(np.logical_and(np.abs(part[:,0,:]-10) < 20, np.abs(part[:,2,:]) < 70))

    ## 3D plot
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*l,linestyle='dashdot',color='orange')
    for (p_i,v_i) in zip(part,v):
        ax.plot(*p_i,alpha=0.7)

    ## projections
    fig, axs = plt.subplots(2,3)
    if isinstance(V,(int,float)):
        fig.suptitle(f'{N} streams with constant velocity {V} at a distance {r} from the curve axis')
    pxy = axs[:,0]
    pyz = axs[:,1]
    pzx = axs[:,2]
    # normalize colormap
    vmin, vmax = (v).min(), (v).max()

    # xy plane
    pxy[0].set_xlabel('x')
    pxy[0].set_ylabel('y')
    pxy[0].plot(*map_xy,color='yellow')
    for (p_i,v_i) in zip(part,v):
        pxy[0].scatter(p_i[0],p_i[1],c=v_i[2],vmin=vmin,vmax=vmax,cmap='RdBu')
    pxy[1].hist(v[ind_z[0],2,ind_z[1]].flatten(),100)
    pxy[1].set_xlabel('$v_z$')
    pxy[1].set_ylabel('counts')

    # yz plane
    pyz[0].set_ylabel('z')
    pyz[0].set_xlabel('y')
    pyz[0].plot(*map_yz,color='yellow')
    for (p_i,v_i) in zip(part,v):
        pyz[0].scatter(p_i[1],p_i[2],c=v_i[0],vmin=vmin,vmax=vmax,cmap='RdBu')
    pyz[1].hist(v[ind_x[0],0,ind_x[1]].flatten(),100)
    pyz[1].set_xlabel('$v_x$')

    # zx plane
    pzx[0].set_xlabel('z')
    pzx[0].set_ylabel('x')
    pzx[0].plot(*map_zx,color='yellow')
    for (p_i,v_i) in zip(part,v):
        pp = pzx[0].scatter(p_i[2],p_i[0],c=v_i[1],vmin=vmin,vmax=vmax,cmap='RdBu')
    cbaxes = fig.add_axes([0.92, 0.1, 0.02, 0.8])  
    fig.colorbar(pp,cax = cbaxes,extend='both',label='$v_i$')
    pzx[1].hist(v[ind_y[0],1,ind_y[1]].flatten(),100)  
    pzx[1].set_xlabel('$v_y$')

    plt.show()

elif test_num == 1:

    display_plot = True

    numpoints = 100
    t = np.linspace(0.3,1.8,numpoints)

    #??
    IND = 42
    SLC = slice(IND,IND+4)
    print('t',t[SLC])

    r = 0.21

    # xl = (t-1)**2/5 + t #10*(t-0.5)**3 + 2*(t-0.5)**2 + (t-0.5) - 2
    # yl = (t-0.5)**2/5 #(t-0.5)**4/6 + 2*(t-0.5)**2 + 1
    # # yl = sin(t*pi) #(t-0.5)**4/6 + 2*(t-0.5)**2 + 1
    # zl = xl/10 + yl*xl
    # xl = t**2
    # yl = t**2
    # zl = t**3

    xl = t
    yl = t**2
    zl = xl + yl


    line = np.array([xl,yl,zl])

    #??
    print('line\n',line[:,SLC])

    # g_xl = 2*(t-1)/5 + 1                              #: dx/dt
    # g_yl = (t-0.5)/5*2                         #: dy/dt
    # # g_yl = cos(t*pi)*pi                         #: dy/dt
    # g_zl = g_xl/10 + (g_xl*yl + xl*g_yl)     #: dz/dt
    # g_xl = 2*t                              #: dx/dt
    # g_yl = 2*t                         #: dy/dt
    # g_zl = 3*t**2     #: dz/dt

    g_xl = np.ones(numpoints)
    g_yl = 2*t
    g_zl = g_xl + g_yl

    m_grad = np.sqrt(g_xl**2+g_yl**2+g_zl**2)
    grad = np.array([g_xl,g_yl,g_zl])/m_grad
    

    # if len(np.where(g_zl==0)[0]) == 0:
    #     x1 = np.ones(numpoints)
    #     y1 = np.ones(numpoints)
    #     z1 = (-abs(g_xl)*x1 -abs(g_yl)*y1)/abs(g_zl) 
    # elif len(np.where(g_xl==0)[0]) == 0:
    #     z1 = np.ones(numpoints)
    #     y1 = np.ones(numpoints)
    #     x1 = (-g_zl*z1 - g_yl*y1)/g_xl
    # elif len(np.where(g_yl==0)[0]) == 0:
    #     z1 = np.ones(numpoints)
    #     x1 = np.ones(numpoints)
    #     y1 = (-g_zl*z1 - g_xl*x1)/g_yl
    # else:
    #     raise

    # x1 = 2 / m_grad
    # y1 = 2 / m_grad
    # z1 = 6*t / m_grad

    x1 = 0 / m_grad
    y1 = 2 / m_grad
    z1 = (x1 + y1) / m_grad

    axis1 = np.array([x1,y1,z1])/np.sqrt(x1**2+y1**2+z1**2)


    # axis2 = np.stack([np.cross(grad[:,i],axis1[:,i]) for i in range(numpoints)],axis=1)
    axis2 = np.stack(np.cross(np.stack(grad,axis=1),np.stack(axis1,axis=1)),axis=1)
    axis2 /= np.sqrt(np.sum(axis2**2,axis=0))
    axis3 = np.stack(np.cross(np.stack(axis2,axis=1),np.stack(grad,axis=1)),axis=1)
    axis3 /= np.sqrt(np.sum(axis3**2,axis=0))

    scal1 = np.array([np.dot(grad[:,i],axis3[:,i]) for i in range(numpoints)])
    scal2 = np.array([np.dot(grad[:,i],axis2[:,i]) for i in range(numpoints)])
    scal3 = np.array([np.dot(axis2[:,i],axis3[:,i]) for i in range(numpoints)])

    print(scal1[scal1!=0])
    print(scal2[scal2!=0])
    print(scal3[scal3!=0])


    def stream_fun(r=3, th0=0):

        mat = [ np.stack(np.array([axis3[:,i],axis2[:,i],grad[:,i]]),axis=1)  for i in range(numpoints)]

        coor_ = np.array([r*cos(th0*pi),r*sin(th0*pi),0])
        coor = np.stack([np.dot(mat[i],coor_) for i in range(numpoints)],axis=1)
        
        #???
        for m in mat[SLC]:
            print('mat\n',m)

        x_,y_,z_ = coor

        x = xl + x_
        y = yl + y_
        z = zl + z_

        #??
        pr_ran = numpoints
        for i in range(pr_ran):
            print(f'{i},{x[i]},{y[i]},{z[i]}')

        return np.array([x,y,z]) 

    stream = stream_fun(r=r)
    # x,y,z = stream

    # X,Y,Z = np.meshgrid(*stream)

    # alpha = np.arctan2(np.sqrt(x**2 + y**2),z)
    # beta  = np.arctan2(y,x)

    # MV = 2*z

    # u = MV * sin(alpha)*cos(beta)
    # v = MV * sin(alpha)*sin(beta)
    # w = MV * cos(alpha)

    # velox = np.array([u,v,w])

    # U,V,W = np.meshgrid(*velox)

    if display_plot:
        P_PARAM = 0

        if P_PARAM == 0:
            import mayavi.mlab as mlab

            # s1 = mlab.flow(*stream,*velox)
            s4 = mlab.plot3d(*line,color=(0,1,0),name='cylinder axis',tube_radius=0.2)
            s5 = mlab.plot3d(*line,color=(0,1,0),name='cylinder axis',tube_radius=None)
            max_th = 1.80
            N = 10
            # mlab.plot3d(*grad,color=(0,0,0))
            # mlab.plot3d(*axis1,color=(1,0,0))
            # mlab.plot3d(*axis2,color=(0,0,1))
            for th0 in np.linspace(0,max_th,N):
                stream = stream_fun(r=r,th0=th0)
                s2 = mlab.plot3d(*stream,color=(th0/max_th,0.5,1))
            # s3 = mlab.quiver3d(*stream,*velox)
            mlab.show()
        else:
            stream = stream_fun(r=r)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot([0,max(xl)],
                    [0,0],
                    [0,0],'--',color='black')
            ax.plot([0,0],
                    [0,max(yl)],
                    [0,0],':',color='black')
            ax.plot([0,0],
                    [0,0],
                    [0,max(zl)],color='black')
            ax.plot(*line,'-.')
            ax.plot(*stream)
            # ax.plot(stream[0,43:45],stream[1,43:45],stream[2,43:45],'.')
            # ax.plot(*grad,'.')
            # ax.plot(*axis1[:,SLC])
            # ax.plot(*axis2[:,SLC])
            fig, (ax1,ax2,ax3) = plt.subplots(1,3)
            ax1.plot(grad[1,IND],grad[2,IND],'or')
            ax1.plot(grad[1],grad[2])
            ax1.plot(axis1[1],axis1[2],'.--')
            ax1.plot(axis2[1],axis2[2],'.--')
            ax2.plot(grad[2,IND],grad[0,IND],'or')
            ax2.plot(grad[2],grad[0])
            ax2.plot(axis1[2],axis1[0],'.--')
            ax2.plot(axis2[2],axis2[0],'.--')
            ax3.plot(grad[0,IND],grad[1,IND],'or')
            ax3.plot(grad[0],grad[1])
            ax3.plot(axis1[0],axis1[1],'.--')
            ax3.plot(axis2[0],axis2[1],'.--')

            plt.show()

elif test_num == 2:

    DISPLAY_PLOT = True

    numpoints = 200
    u = np.linspace(-0.2,1.2,numpoints)

    x_i = (u+1)**2
    x_j = sin(u*pi)
    x_k = x_i * x_j

    x = np.array([x_i,x_j,x_k])

    t_i = 2*(u+1)
    t_j = cos(u*pi)*pi
    t_k = t_i * x_j + x_i * t_j

    t = np.array([t_i,t_j,t_k])/np.sqrt(t_i**2+t_j**2+t_k**2)

    r = np.array([[1],[0],[-t_i[0]/t_k[0]]])
    r /= np.sqrt(np.sum(r**2,axis=0))

    s = np.cross(t[:,0],r[:,0]).reshape(3,1)
    s /= np.sqrt(np.sum(s**2,axis=0))


    r,s,t = alg.double_reflection(x,t,r,s,numpoints)

    R = 0.2
    
    def compute_stream(R: float = R,th0: float = 0) -> NDArray:
        stream = np.array([R*cos(u*pi + th0*pi),R*sin(u*pi + th0*pi),np.zeros(numpoints)])
        mat = [np.array([r[:,ui],s[:,ui],t[:,ui]]).T for ui in range(numpoints) ]

        return np.array([ np.dot(mat[ui],stream[:,ui]) for ui in range(numpoints)]).T + x

    print('num  \tr · s\tr · t\tt · s')
    for ui in range(numpoints):
        print(f'<{ui:3d}>\t{abs(np.dot(r[:,ui],s[:,ui])):.5f}\t{abs(np.dot(r[:,ui],t[:,ui])):.5f}\t{abs(np.dot(t[:,ui],s[:,ui])):.5f}')

    print('\n- - - STREAM - - -')    
    stream1 = compute_stream()
    stream2 = compute_stream(th0=1.5)

    print('num  \ts1 · s2')
    for ui in range(numpoints):
        print(f'({ui:3d})\t{np.dot(stream1[:,ui],stream2[:,ui]):.5f}')
    
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