import numpy as np
from numpy import pi
from numpy.typing import NDArray
import matplotlib.pyplot as plt

class RS0():
    def __init__(self, x: float | NDArray = 0., y: float | NDArray = 0., z: float | NDArray = 0.) -> None:
        self.x = x
        self.y = y
        self.z = z        

    def coord(self) -> tuple:
        return self.x, self.y, self.z

class RS1():
    def __init__(self, r: float | NDArray = 0., theta: float | NDArray = 0., z: float | NDArray = 0.) -> None:
        self.r = r
        self.theta = theta
        self.z = z

    def cartesian(self) -> tuple[float | NDArray, float | NDArray]:
        r, theta = self.r, self.theta
        x = r*np.cos(theta)        
        y = r*np.sin(theta)
        return x, y        

    def coord(self, mode: int = 0) -> tuple:
        if mode == 0:
            x,y = self.cartesian()
            return x, y, self.z
        else:
            return self.r, self.theta, self.z

        

def torque(axis: str, Rc: float, ang: float, point: RS0):
    if axis == 'z':
        x_0, y_0, _ = point.coord()

if __name__ == '__main__':
    x_line = lambda k : np.sin(k*2*pi)*np.exp(k)
    y_line = lambda k : 1/(k+1)
    z_line = lambda k : k**3 

    k = np.linspace(0,2,100)
    x1_0 = x_line(k)
    y1_0 = y_line(k)
    z1 = z_line(k)
    theta1 = np.linspace(0,2*pi,100)
    r1 = 5 

    Theta, K = np.meshgrid(theta1, k)
    X1_0 = x_line(K)
    Y1_0 = y_line(K)
    Z = z_line(K)
    X = r1*np.cos(Theta) + X1_0
    Y = r1*np.sin(Theta) + Y1_0

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot_surface(X,Y,Z,alpha=0.6)
    ax.plot(x1_0,y1_0,z1)

    # ax.set_aspect('equal')

    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(Y,Z,color='blue',alpha=0.6)
    plt.plot(y1_0,z1,color='orange')
    plt.subplot(1,3,2)
    plt.plot(X,Z,color='blue',alpha=0.6)
    plt.plot(x1_0,z1,color='orange')
    plt.subplot(1,3,3)
    plt.plot(X,Y,color='blue',alpha=0.6)
    plt.plot(x1_0,y1_0,color='orange')

    plt.show()