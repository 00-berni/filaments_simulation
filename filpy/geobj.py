from typing import Any, Sequence
import numpy as np
from numpy import pi, sin, cos, ndarray
from numpy.typing import NDArray
import matplotlib.pyplot as plt


ID = np.array([[1., 0., 0.],    #: identity matrix
               [0., 1., 0.],
               [0., 0., 1.]])


def rot_mat(ang: float | int, axis: int) -> NDArray:
    """To compute the rotation matrix

    The unit of angle value has to be rad/pi
    
    Parameters
    ----------
    ang : float | int
        rotation angle value [rad/pi]
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
    rmat = ID.copy()
    # compute sin and cos
    s,c = sin(ang*pi), cos(ang*pi)
    # extract the axis of rotation plane
    ind_axis = [0,1,2]
    ind_axis.remove(axis)
    m, n = ind_axis
    rmat[[m,n],[m,n]] = np.array([c,c])
    rmat[[m,n],[n,m]] = np.array([-s,s])*(1-2*(axis % 2))
    return rmat

def compute_rotation(ang: int | float | Sequence[int | float], axis: str) -> NDArray:
    """To compute the rotation matrix 

    It is possible to combine rotations around different axes:
        
      * pass a list (or a tuple) for `ang` with different rotation angles
      * pass a string for `axis` with the names of the rotation axes  

    The unit of angle value(s) has to be rad/pi

    Parameters
    ----------
    ang : int | float | Sequence[int  |  float]
        rotation angle value(s) [rad/pi]
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
    rmat = ID.copy()
    # check the angle type
    if len(axis) == 1: ang = [ang]
    # compute the combination of the rotation matrices
    for ax in axis:
        pos = axis.find(ax)
        if ang[pos] != 0:
            # compute the combination
            rmat = rot_mat(ang[pos],indx[ax]) @ rmat
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
    param : None | NDArray
        curve parameter values
    line : None | NDArray
        spine curve vector
    r : None | NDArray
        normal unit vector
    s : None | NDArray
        unit vector obtain by cross product
    t : None | NDArray
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
    EMPTY_FRAME = (None,None,None,None,None)

    @staticmethod
    def vec_module(vec: NDArray) -> float:
        """To compute the module of a vectors collection

        Parameters
        ----------
        vec : NDArray
            selected vectors collection

        Returns
        -------
        float
            collection of their modules
        """
        return np.sqrt(np.sum(vec**2,axis=0))

    @staticmethod
    def normalize(vec: NDArray) -> NDArray:
        """To normalize a vector

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
        return vec/mod

    @staticmethod
    def double_reflection(x: NDArray, t: NDArray, r: NDArray, s: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """To compute the frame along the spine curve

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

    def __init__(self, u: NDArray | None, line_eq: NDArray | None, t: NDArray | None, r0: NDArray | None, s0: NDArray | None) -> None:
        """Constructor of the class

        Parameters
        ----------
        u : NDArray
            values of the spine curve parameter
        line_eq : NDArray
            spine curve
        t : NDArray
            tangent vector
        r0 : NDArray
            initial value for normal vector
        s0 : NDArray
            initial value for third vector
        """
        # create an empty object
        if u is None:
            self.param = u
            self.line = line_eq
            self.r = r0
            self.s = s0
            self.t = t
        else:
            # check if the tangent vector is normalized
            if np.any(Frame.vec_module(t) != 1): t = Frame.normalize(t)
            # compute the frame along the spine curve
            r,s,t = Frame.double_reflection(x=line_eq,t=t,r=r0,s=s0)
            
            self.param = u.copy()
            self.line = line_eq.copy()
            self.r = r.copy()
            self.s = s.copy()
            self.t = t.copy()
    
        
    def frame_matrices(self) -> NDArray:
        """To compute the matrices to change reference system

        The method compute the matrix to pass from the frame attached 
        to the spine curve to the cartesian one for each point of
        the collection. 
        
        The matrix is simply:

            [[r_i, s_i, t_i]
             [r_j, s_j, t_j]
             [r_k, s_k, t_k]]

        Returns
        -------
        NDArray
            collection of transformation matrices
        """
        numpoints = len(self.param)     #: number of points of the collection
        r = self.r
        s = self.s
        t = self.t
        return np.array([np.array([r[:,ui],s[:,ui],t[:,ui]]).T for ui in range(numpoints)])

    def copy(self):
        new_frame = Frame(*Frame.EMPTY_FRAME)
        new_frame.param = self.param.copy()
        new_frame.line = self.line.copy()
        new_frame.r = self.r.copy()
        new_frame.s = self.s.copy()
        new_frame.t = self.t.copy()
        return new_frame

    def rotate(self, ang: int | float | Sequence[int | float], axis: str):
        """To rotate the frame coordinates

        It is possible to combine rotations around different axes:

          * pass a list (or a tuple) for ang with different rotation angles
          * pass a string for axis with the names of the rotation axes

        The unit of angle value(s) has to be rad/pi
    
        Parameters
        ----------
        ang : int | float | Sequence[int  |  float]
             rotation angle value(s) [rad/pi]
        axis : str
            names of the rotation axis(es)
        """
        # compute the rotation matrix
        rmat = compute_rotation(ang = ang, axis = axis)
        numpoints = len(self.param)     #: number of points  of the collection
        new_frame: Frame = self.copy()
        # compute the rotated vectors
        new_frame.line = np.array([rmat @ new_frame.line[:,ui] for ui in range(numpoints)]).T
        new_frame.r    = Frame.normalize(np.array([rmat @ new_frame.r[:,ui] for ui in range(numpoints)]).T)
        new_frame.s    = Frame.normalize(np.array([rmat @ new_frame.s[:,ui] for ui in range(numpoints)]).T)
        new_frame.t    = Frame.normalize(np.array([rmat @ new_frame.t[:,ui] for ui in range(numpoints)]).T)
        return new_frame

class StreamLine():
    """Class collects all the information about the 
    trajectory and velocity of a line of the tube

    Attributes
    ----------
    par : dict
        parameters of the trajectory line
    pos : None | NDArray
        trajectory coordinates
    vel : None | NDArray
        velocity values along the trajectory
    """
    def __init__(self, omega: int | float, R: int | float, th0: int | float) -> None:
        """Constructor of the class

        Parameters
        ----------
        omega : int | float
            wrapping of the trajectory line
        R : int | float
            distance from the spine curve
        th0 : int | float
            initial angle relative to the unit vector `r`
        """
        self.par = {'omega': omega, 'R': R, 'th0': th0}
        self.pos = None
        self.vel = None

    def compute_stream(self, frame: Frame, v: float | NDArray | None, field: str = 'T'):
        """To compute the trajectory and the velocity of a line
        of the filament along the spine curve

        A constant value of `v` is considered as a constant 
        velocity vector along `t` direction

        Parameters
        ----------
        frame : Frame
            frame attached to the spine curve
        v : float | NDArray | None
            velocity along the trajectory in addition to
            rotational velocity (`R*omega`)
        field : str
            if `True`,`v` is considered the module of a 
            uniform velocity field
        Returns
        -------
        StreamLine
            the computed filament line
        """
        # extract parametrs of the line
        omega, R, th0 = self.par.values()
        x = frame.line.copy()      #: spine curve
        # compute the transformation matrix along `x`
        frame_mat = frame.frame_matrices()
        numpoints = len(frame.param)        #: number of points of the collection
        ui = np.linspace(0,2,numpoints)     #: wrapping parametrization
        
        ## Trajectory
        # compute the trajectory coordinates in the frame attached to `x`
        stream = np.array([R*cos(omega*pi*ui + th0*pi),R*sin(omega*pi*ui + th0*pi),np.zeros(numpoints)])
        # compute the cartesian coordinates of the trajectory
        self.pos = np.array([ frame_mat[ui] @ stream[:,ui] for ui in range(numpoints)]).T + x

        ## Velocity
        if v is None:
            v = 0
        # for the uniform field v => v_t
        if field[:4] == 'unif' and omega != 0:
            if np.any(v**2 < (omega*pi*R)**2):
                raise Exception('\tv**2 - (omega*R)**2 < 0\nChange `omega` or `v` value!')
            # tangent component to have constant module
            v = np.sqrt(v**2 - (omega*pi*R)**2)
        # check the type of `v`
        if isinstance(v,(int,float)): 
            v = np.array([np.zeros(numpoints),np.zeros(numpoints),np.full(numpoints,v)])
        elif type(v) != ndarray and field != 'T':
            direc = {'r': 0, 's': 1, 't': 2}
            new_v = np.zeros(stream.shape)
            indx = [direc[d] for d in field]
            var = [stream[:,ix] for ix in indx]
            new_v[:,2] = v(*var)
            v = new_v
        velox = v.copy()
        if omega != 0:
            # compute the velocity coordinates in the frame attached to `x`
            velox += np.array([-omega*R*pi*sin(omega*ui*pi + th0*pi),omega*R*pi*cos(omega*ui*pi + th0*pi),np.zeros(numpoints)]) 
        # compute the cartesian coordinates of the velocity
        self.vel = np.array([ frame_mat[ui] @ velox[:,ui] for ui in range(numpoints)]).T + frame.t

        return self

    
    def copy(self):
        """To copy a frame

        Returns
        -------
        StreamLine
            the copy of the frame
        """
        new_line = StreamLine(**self.par)
        new_line.pos = self.pos.copy()
        new_line.vel = self.vel.copy()
        return new_line

    def rotate(self, ang: int | float | Sequence[int | float], axis: str):
        """To rotate the filament line

        It is possible to combine rotations around different axes:

            * pass a list (or a tuple) for ang with different rotation angles
            * pass a string for axis with the names of the rotation axes
        
        The unit of angle value(s) has to be rad/pi

        Parameters
        ----------
        ang : int | float | Sequence[int  |  float]
            rotation angle value(s) [rad/pi]
        axis : str
            names of the rotation axis(es)
        """
        # compute the rotation matrix
        rmat = compute_rotation(ang, axis)
        numpoints = self.pos.shape[-1]       #: number of points of the collection
        new_stream: StreamLine = self.copy()
        # compute the rotated vectors
        new_stream.pos = np.array([ rmat @ new_stream.pos[:,ui] for ui in range(numpoints) ]).T
        new_stream.vel = np.array([ rmat @ new_stream.vel[:,ui] for ui in range(numpoints) ]).T
        return new_stream

class Filament():

    EMPTY_FIL = (Frame(*Frame.EMPTY_FRAME),None,None,None)

    """Class collects information about filament lines

    Attributes
    ----------
    frame : Frame
        the reference frame attached to the spine curve
    lines : list[StreamLine]
        list of all lines of the filament
    trj : NDArray
        array with the trajectories of the lines
    vel : NDArray
        array with the velocities of the lines
    """
    def __init__(self, frame: Frame, line_param: Sequence[float | NDArray] | None, v: float | NDArray | None, field: str | None) -> None:
        """Constructor of the class

        Parameters
        ----------
        frame : Frame
            reference frame attached to the spine curve of the filament
        line_param : Sequence[float  |  NDArray]
            parameters of the line, like (omega, radius, theta0)
        v : float | NDArray
            velocity field in addition to rotation
        unif_field : bool, default False
            if `True`,`v` is considered the module of a 
            uniform velocity field 
        """
        if frame.t is None:
            self.frame = frame
            self.lines = None
            self.trj = None
            self.vel = None
        else:
            omega, R, th0 = line_param
            collection = [StreamLine(omega, Ri, thi).compute_stream(frame, v, field=field) for Ri in R for thi in th0]

            self.frame = frame.copy()
            self.lines = [line.copy() for line in collection]
            self.trj = np.array([line.pos for line in collection])
            self.vel = np.array([line.vel for line in collection])
    
    def copy(self):
        new_filament = Filament(*Filament.EMPTY_FIL)
        new_filament.frame = self.frame.copy()
        new_filament.lines = [line.copy() for line in self.lines]
        new_filament.trj = self.trj.copy()
        new_filament.vel = self.vel.copy()
        return new_filament

    def rotate(self, ang: int | float | Sequence[int | float], axis: str):
        new_filament: Filament = self.copy()
        new_filament.frame = new_filament.frame.rotate(ang, axis)
        new_filament.lines = [line.rotate(ang, axis) for line in new_filament.lines]
        new_filament.trj = np.array([line.pos for line in new_filament.lines])
        new_filament.vel = np.array([line.vel for line in new_filament.lines])
        return new_filament
