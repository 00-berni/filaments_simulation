from typing import Any, Sequence
import numpy as np
from numpy import pi, sin, cos
from numpy.typing import NDArray
import matplotlib.pyplot as plt


ID = np.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])


def rot_mat(ang: float | int, axis: int) -> NDArray:
    """Function to compute the rotation matrix

    Parameters
    ----------
    ang : float | int
        rotation angle value
    axis : int
        axis respect to which system rotates 
        (`[0,1,2]` --> `[i,j,k]`)

    Returns
    -------
    NDArray
        matrix of rotation
        (if `ang = 0` then it returs identity)
    """
    # initialize rotation matrix as identity
    rmat = ID
    # compute sin and cosfloat
    s,c = sin(ang*pi), cos(ang*pi)
    # extract the axis of rotation plane
    m, n = [0,1,2].remove(axis)
    rmat[[m,n],[m,n]] = np.array([c,c])
    rmat[[m,n],[n,m]] = np.array([-s,s])*(-(axis % 2))
    return rmat

def compute_rotation(ang: int | float | Sequence[int | float], axis: str) -> NDArray:
    """Function to compute the rotation matrix 

    It is possible to combine rotations around different axes:
        
      * pass a list (or a tuple) for `ang` with different rotation angles
      * pass a string for `axis` with the names of the rotation axes  

    Parameters
    ----------
    ang : int | float | Sequence[int  |  float]
        rotation angle value(s)
    axis : str
        names of the rotation axis(es)

    Returns
    -------
    NDArray
        rotation matrix or a combination of ones
    """
    # define a dictionary to convert axis name in index
    indx = {'i': 0, 'j': 1, 'k': 2}
    # initialize rotation matrix as identity
    rmat = ID
    # check the angle type
    if len(axis) == 1: ang = [ang]
    # compute the combination of the rotation matrices
    for ax in axis:
        pos = axis.find(ax)
        # compute the combination
        rmat = rmat @ rot_mat(ang[pos],indx[ax])
    return rmat



class Frame():
    """Class stores information about the orthonormal frame
    attached to a spine curve `x(u)`.

    The orthonormal frame is composed by 
    three unit vectors (r,s,t):

        * `t = dx/du / ||dx/du||`
        * `r` : `r · t = 0`
        * `s = t x r`

    Attributes
    ----------
    param : None | NDarray
        curve parameter values
    line : None | NDarray
        spine curve vector
    r : None | NDarray
        normal unit vector
    s : None | NDarray
        unit vector obtain by cross product
    t : None | NDarray
        tangent unit vector of `line`
    
    Methods
    -------
    vec_module(vec)                 @staticmethod
        compute the module of a vectors collection
    normalize(vec)                  @staticmethod
        normalize a vector
    double_reflection(x, t, r, s)   @staticmethod
        compute the frame along the spine curve
    __init__(u, line_eq, t, r0, s0)
        constructor of the class
    rotate(ang, axis)
        rotate the frame
    frame_matricies()
        compute the matrices to change reference system
    """
    @staticmethod
    def vec_module(vec: NDArray) -> float:
        """Function to compute the module of
        a vectors collection

        Parameters
        ----------
        vec : NDArray
            selected vectors collection

        Returns
        -------
        float
            collection of their modules
        """
        return np.sum(vec**2,axis=0)

    @staticmethod
    def normalize(vec: NDArray) -> NDArray:
        """Function to normalize a vector

        Parameters
        ----------
        vec : NDArray
            selected vectors collection

        Returns
        -------
        NDArray
            normalized vectors collection
        """
        mod = Frame.vec_module(vec) 
        return vec / mod

    @staticmethod
    def double_reflection(x: NDArray, t: NDArray, r: NDArray, s: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Function to compute the frame along the spine curve

        The method requires the spine curve (`x`), its tangent vector (`t`) 
        and some initial values for the normal (`r`) and third (`s`) vectors
        as input

        The algorithm is taken from [1]_ 

        Parameters
        ----------
        x : NDArray
            spine curve values
        t : NDArray
            tangent vector values
        r : NDArray
            initial vector for normal vector
        s : NDArray
            initial vector for third vector

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            the ordered triple (`r`,`s`,`t`) along `x`


        References
        ----------
        .. [1] Wang, W., Jüttler, B., Zheng, D., and Liu, Y. 2008. Computation of rotation minimizing frame. 
               ACM Trans. Graph. 27, 1, Article 2 (March 2008), 18 pages.
               DOI = 10.1145/1330511.1330513
               URL = http://doi.acm.org/10.1145/1330511.1330513
        """
        n = x.shape[-1]     #: number of points
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

    def __init__(self, u: NDArray, line_eq: NDArray, t: NDArray, r0: NDArray, s0: NDArray) -> None:
        if np.any(Frame.vec_module(t) != 1): t = Frame.normalize(t)
        r,s,t = Frame.double_reflection(x=line_eq,t=t,r=r0,s=s0)
        
        self.param = u.copy()
        self.line = line_eq.copy()
        self.r = r.copy()
        self.s = s.copy()
        self.t = t.copy()
    
    def rotate(self, ang: int | float | Sequence[int | float], axis: str) -> None:
        rmat = compute_rotation(ang = ang, axis = axis)
        numpoints = len(self.param)
        self.line = Frame.normalize(np.array([rmat @ self.line[:,ui] for ui in range(numpoints)]))
        self.r    = Frame.normalize(np.array([rmat @ self.r[:,ui] for ui in range(numpoints)]))
        self.s    = Frame.normalize(np.array([rmat @ self.s[:,ui] for ui in range(numpoints)]))
        self.t    = Frame.normalize(np.array([rmat @ self.t[:,ui] for ui in range(numpoints)]))
        
    def frame_matrices(self) -> NDArray:
        numpoints = len(self.param) 
        r = self.r
        s = self.s
        t = self.t
        return np.array([np.array([r[:,ui],s[:,ui],t[:,ui]]).T for ui in range(numpoints)])


class StreamLine():
    
    def __init__(self, omega: int | float, R: int | float, th0: int | float) -> None:
        
        self.par = {'w': omega, 'R': R, 'th0': th0}

        self.pos = None
        self.vel = None

    def compute_stream(self, frame: Frame, vT: float | NDArray):
        omega, R, th0 = self.par.values()
        x = frame.line
        frame_mat = frame.frame_matrices()
        numpoints = len(frame.param)
        ui = np.linspace(0,2,numpoints)
        
        # compute the streamline in the frame of the curve
        stream = np.array([R*cos(omega*ui*pi + th0*pi),R*sin(omega*ui*pi + th0*pi),np.zeros(numpoints)])
        # compute the cartesian coordinates of the streamline
        self.pos = np.array([ frame_mat[ui] @ stream[:,ui] for ui in range(numpoints)]).T + x

        if isinstance(vT,(int,float)): 
            vT = np.full(numpoints,vT)
        # compute the streamline in the frame of the curve
        velox = np.array([-omega*R*pi*sin(omega*ui*pi + th0*pi),omega*R*pi*cos(omega*ui*pi + th0*pi),vT])
        # compute the cartesian coordinates of the streamline
        self.vel = np.array([ frame_mat[ui] @ velox[:,ui] for ui in range(numpoints)]).T 

        return self

    def rotate(self, ang: int | float | Sequence[int | float], axis: str) -> None:
        rmat = compute_rotation(ang, axis)
        numpoints = len(self.param)
        self.pos = np.array([ rmat @ self.pos[:,ui] for ui in range(numpoints)])
        self.vel = np.array([ rmat @ self.vel[:,ui] for ui in range(numpoints)])

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
