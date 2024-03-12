from typing import Any
import numpy as np
from numpy import pi, sin, cos
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def double_reflection(x:NDArray,t:NDArray,r:NDArray,s:NDArray,n:int) -> tuple[NDArray, NDArray, NDArray]:
    for i in range(n-1):
        # compute reflection vector of R1.
        v1 = x[:,i+1] - x[:,i]
        c1 = np.dot(v1,v1)
        # compute rL_i = R1 r_i
        rL = r[:,i] - (2/c1 ) * np.dot(v1, r[:,i]) * v1
        # compute tL_i = R1 t_i 
        tL = t[:,i] - (2/c1 ) * np.dot(v1, t[:,i]) * v1 
        # compute reflection vector of R2
        v2 = t[:,i+1] - tL
        c2 = np.dot(v2, v2)
        # compute ri+1 = R2rL_i 
        r = np.append(r, (rL - (2/c2 ) * np.dot(v2, rL) * v2).reshape(3,1),axis=1) 
        # compute vector si+1 of Ui+1
        s = np.append(s, np.cross(t[:,i+1], r[:,i+1]).reshape(3,1), axis=1)
    return r,s,t

def mat_to_arr(mat: NDArray, axis: int) -> NDArray:
    return mat[:,axis].reshape(3,1)

def arr_to_mat(*args) -> NDArray:
    return np.concatenate(args,axis=1)
    

# Std Basis
X0 = np.array([[1],[0],[0]])
Y0 = np.array([[0],[1],[0]])
Z0 = np.array([[0],[0],[1]])

ID = arr_to_mat(X0,Y0,Z0)

def rot_mat(axis: int, ang: float) -> NDArray:
    c = cos(ang)
    s = sin(ang)
    mat = ID.copy().astype(float)
    idxs = [0,1,2]
    idxs.remove(axis)
    idx1 = idxs*2
    idx2 = idxs+idxs[::-1]
    
    mat[idx1,idx2] = np.array([c,c,-s,s])
    return mat

class AxSys():
    def __init__(self, x1: NDArray = X0, y1: NDArray = Y0, z1: NDArray = Z0 ) -> None:
        self.x1 = np.array([x1])
        self.y1 = np.array([y1])
        self.z1 = np.array([z1])

    def coor(self) -> tuple[NDArray,NDArray,NDArray]:
        return self.x1, self.y1, self.z1

    def matrix_pos(self) -> NDArray:
        x,y,z = self.coor()
        return np.array([arr_to_mat(i,j,k) for (i,j,k) in zip(x,y,z)])


class Point():
    def __init__(self) -> None:
        pass




# if __name__ == '__main__':
    
#     # direction line 
#     t = np.linspace(0,100,1000)
#     xl = np.zeros(len(t)) + 50
#     yl = np.zeros(len(t)) - 10
#     zl = t

#     # cylider
#     r = 5
#     theta0 = pi/4
    
#     z = zl
#     x = r*np.cos(z+theta0) + xl
#     y = r*np.sin(z+theta0) + yl

#     # direction line 
#     xl2 = t
#     yl2 = np.ones(len(t)) * 5
#     zl2 = np.zeros(len(t)) + 50

#     # cylider
#     # r = 3
#     # theta0 = pi/4
    
#     x2 = xl2
#     z2 = r*np.cos(x2+theta0) + zl2
#     y2 = r*np.sin(x2+theta0) + yl2

#     ax = plt.figure().add_subplot(projection='3d')
#     ax.plot(x,y,z,color='blue')
#     ax.plot(x2,y2,z2,color='green')
#     # ax.plot(xl,yl,zl,color='orange')
#     # ax.plot(xl2,yl2,zl2,color='red')

#     plt.figure()
#     plt.subplot(131)
#     plt.title('x-y')
#     plt.plot(x,y,color='blue')
#     plt.plot(x2,y2,color='red')
#     # plt.plot(xl,yl,color='orange')
#     # plt.plot(xl2,yl2,color='green')
#     plt.subplot(132)
#     plt.title('y-z')
#     plt.plot(y,z,color='blue')
#     plt.plot(y2,z2,color='red')
#     # plt.plot(yl,zl,color='orange')
#     # plt.plot(yl2,zl2,color='green')
#     plt.subplot(133)
#     plt.title('z-x')
#     plt.plot(z,x,color='blue')
#     plt.plot(z2,x2,color='red')
#     # plt.plot(zl,xl,color='orange')
#     # plt.plot(zl2,xl2,color='green')
#     plt.show()
