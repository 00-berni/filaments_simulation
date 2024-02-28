import numpy as np
from numpy import pi
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# def stream_line(tmin: int, tmax: int, numpoints: int, )



if __name__ == '__main__':
    
    # direction line 
    t = np.linspace(0,100,1000)
    xl = np.zeros(len(t)) + 20
    yl = np.zeros(len(t)) - 10
    zl = t

    # cylider
    r = 5
    theta0 = pi/4
    
    z = zl
    x = r*np.cos(z+theta0) + xl
    y = r*np.sin(z+theta0) + yl

    # direction line 
    xl2 = t
    yl2 = np.ones(len(t)) * 5
    zl2 = np.zeros(len(t)) + 50

    # cylider
    # r = 3
    # theta0 = pi/4
    
    x2 = xl2
    z2 = r*np.cos(x2+theta0) + zl2
    y2 = r*np.sin(x2+theta0) + yl2

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x,y,z,color='blue')
    ax.plot(x2,y2,z2,color='green')
    # ax.plot(xl,yl,zl,color='orange')
    # ax.plot(xl2,yl2,zl2,color='red')

    plt.figure()
    plt.subplot(131)
    plt.title('x-y')
    plt.plot(x,y,color='blue')
    plt.plot(x2,y2,color='red')
    # plt.plot(xl,yl,color='orange')
    # plt.plot(xl2,yl2,color='green')
    plt.subplot(132)
    plt.title('y-z')
    plt.plot(y,z,color='blue')
    plt.plot(y2,z2,color='red')
    # plt.plot(yl,zl,color='orange')
    # plt.plot(yl2,zl2,color='green')
    plt.subplot(133)
    plt.title('z-x')
    plt.plot(z,x,color='blue')
    plt.plot(z2,x2,color='red')
    # plt.plot(zl,xl,color='orange')
    # plt.plot(zl2,xl2,color='green')
    plt.show()
