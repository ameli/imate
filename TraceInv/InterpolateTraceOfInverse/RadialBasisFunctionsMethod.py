# =======
# Imports
# =======

from __future__ import print_function
from .InterpolantBaseClass import InterpolantBaseClass

import numpy
import scipy
from scipy import interpolate

# =============================
# Radial Basis Functions Method
# =============================

class RadialBasisFunctionsMethod(InterpolantBaseClass):
    """
    Computes the trace of inverse of an invertible matrix :math:`\\mathbf{A} + t \\mathbf{B}` using 
    an interpolation scheme based on rational polynomial functions (see details below).

    **Class Inheritance:**

    .. inheritance-diagram:: TraceInv.InterpolateTraceOfInverse.RadialBasisFunctionsMethod
        :parts: 1

    :param A: Invertible matrix, can be either dense or sparse matrix.
    :type A: numpy.ndarray

    :param B: Invertible matrix, can be either dense or sparse matrix.
    :type B: numpy.ndarray

    :param ComputeOptions: A dictionary of input arguments for :mod:`TraceInv.ComputeTraceOfInverse.ComputeTraceOfInverse` module.
    :type ComputeOptions: dict

    :param Verbose: If ``True``, prints some information on the computation process. Default is ``False``.
    :type Verbose: bool

    :param FunctionType: Can be ``1``, ``2``, or ``3``, which defines different radial basis functions (see details below).
    :type FunctionType: int

    **Interpolation Method**
    
    Define the function

    .. math::

        \\tau(t) = \\frac{\\mathrm{trace}\\left( (\\mathbf{A} + t \\mathbf{B})^{-1} \\right)}{\mathrm{trace}(\mathbf{B}^{-1})}

    and :math:`\\tau_0 = \\tau(0)`. Then, we approximate :math:`\\tau(t)` by radial basis functions as follows. Define

    .. math::

        x(t) = \log t

    Depending whether ``FunctionType`` is set to ``1``, ``2``, or ``3``, one of the following functions is defined:

    .. math::
        :nowrap:

        \\begin{eqnarray}
        y_1(t) &= \\frac{1}{\\tau(t)} - \\frac{1}{\\tau_0} - t, \\\\
        y_2(t) &= \\frac{\\frac{1}{\\tau(t)}}{\\frac{1}{\\tau_0} + t} - 1, \\\\
        y_3(t) &= 1 - \\tau(t) \left( \\frac{1}{\\tau_0} + t \\right).
        \end{eqnarray}

    * The set of data :math:`(x,y_1(x))` are interpolated using *cubic splines*.
    * The set of data :math:`(x,y_2(x))` and :math:`(x,y_3(x))` are interpolated using *Gaussian radial basis functions*.

    **Example**

    This class can be invoked from :class:`TraceInv.InterpolateTraceOfInverse.InterpolateTraceOfInverse` module 
    using ``InterpolationMethod='RBF'`` argument.

    .. code-block:: python

        >>> from TraceInv import GenerateMatrix
        >>> from TraceInv import InterpolateTraceOfInverse
        
        >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
        >>> A = GenerateMatrix(NumPoints=20)
        
        >>> # Create an object that interpolates trace of inverse of A+tI (I is identity matrix)
        >>> TI = InterpolateTraceOfInverse(A,InterpolatingMethod='RBF')
        
        >>> # Interpolate A+tI at some input point t
        >>> t = 4e-1
        >>> trace = TI.Interpolate(t)
    """

    # ----
    # Init
    # ----

    def __init__(self,A,B=None,InterpolantPoints=None,ComputeOptions={},Verbose=False,FunctionType=1):
        """
        Initializes the base class and attributes.
        """

        # Base class constructor
        super(RadialBasisFunctionsMethod,self).__init__(A,B=B,InterpolantPoints=InterpolantPoints,
                ComputeOptions=ComputeOptions,Verbose=Verbose)

        # Initialize Interpolator
        self.RBF = None
        self.LowLogThreshold = None
        self.HighLogThreshold = None
        self.FunctionType = FunctionType
        self.InitializeInterpolator()

    # -----------------------
    # Initialize Interpolator
    # -----------------------

    def InitializeInterpolator(self):
        """
        Finds the coefficients of the interpolating function.
        """
        
        if self.Verbose:
            print('Initialize interpolator ...')
        
        # Take logarithm of t_i
        xi = numpy.log10(self.t_i)

        if xi.size > 1:
            dxi = numpy.mean(numpy.diff(xi))
        else:
            dxi = 1

        # Function Type
        if self.FunctionType == 1:
            # Ascending function
            yi = 1.0/self.tau_i - (1.0/self.tau0 + self.t_i)
        elif self.FunctionType == 2:
            # Bell shape, going to zero at boundaries
            yi = (1.0/self.tau_i)/(1.0/self.tau0 + self.t_i) - 1.0
        elif self.FunctionType == 3:
            # Bell shape, going to zero at boundaries
            yi = 1.0 - (self.tau_i)*(1.0/self.tau0 + self.t_i)
        else:
            raise ValueError('Invalid function type.')

        # extend boundaries to zero
        self.LowLogThreshold = -4.5   # SETTING
        self.HighLogThreshold = 3.5   # SETTING
        NumExtend = 3                 # SETTING
       
        # Avoid thresholds to cross interval of data
        if self.LowLogThreshold >= numpy.min(xi):
            self.LowLogThreshold = numpy.min(xi) - dxi
        if self.HighLogThreshold <= numpy.max(xi):
            self.HighLogThreshold = numpy.max(xi) + dxi

        # Extend interval of data by adding zeros to left and right
        if (self.FunctionType == 2) or (self.FunctionType == 3):
            ExtendLeft_x = numpy.linspace(self.LowLogThreshold-dxi,self.LowLogThreshold,NumExtend)
            ExtendRight_x = numpy.linspace(self.HighLogThreshold,self.HighLogThreshold+dxi,NumExtend)
            Extend_y = numpy.zeros(NumExtend)
            xi = numpy.r_[ExtendLeft_x,xi,ExtendRight_x]
            yi = numpy.r_[Extend_y,yi,Extend_y]

        # Radial Basis Function
        if self.FunctionType == 1:
            # These interpolation methods are good for the ascending shaped function
            self.RBF = scipy.interpolate.CubicSpline(xi,yi,bc_type=((1,0.0),(2,0)),extrapolate=True)       # best for ascending function
            # self.RBF = scipy.interpolate.PchipInterpolator(xi,yi,extrapolate=True)                       # good
            # self.RBF = scipy.interpolate.UnivariateSpline(xi,yi,k=3,s=0.0)                               # bad
        elif (self.FunctionType == 2) or (self.FunctionType == 3):
            # These interpolation methods are good for the Bell shaped function
            self.RBF = scipy.interpolate.Rbf(xi,yi,function='gaussian',epsilon=dxi)                    # Best for function type 2,3,4
            # self.RBF = scipy.interpolate.Rbf(xi,yi,function='inverse',epsilon=dxi)
            # self.RBF = scipy.interpolate.CubicSpline(xi,yi,bc_type=((1,0.0),(1,0.0)),extrapolate=True)

        # Plot interpolation with RBF
        # PlotFlag = False
        # if PlotFlag:
        #     import matplotlib.pyplot as plt
        #     t = numpy.logspace(self.LowLogThreshold-dxi,self.HighLogThreshold+dxi,100)
        #     x = numpy.log10(t)
        #     y = self.RBF(x)
        #     fig,ax = plt.subplots()
        #     ax.plot(x,y)
        #     ax.plot(xi,yi,'o')
        #     ax.grid(True)
        #     ax.set_xlim([self.LowLogThreshold-dxi,self.HighLogThreshold+dxi])
        #     # ax.set_ylim(-0.01,0.18)
        #     plt.show()

        if self.Verbose:
            print('Done.')

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t):
        """
        Interpolates :math:`\mathrm{trace} \left( (\mathbf{A} + t \mathbf{B})^{-1} \\right)` at :math:`t`.

        This is the main interface function of this module and it is used after the interpolation
        object is initialized.

        :param t: The inquiry point(s).
        :type t: float, list, or numpy.array

        :return: The interpolated value of the trace.
        :rtype: float or numpy.array
        """
        
        x = numpy.log10(t)
        if (x < self.LowLogThreshold) or (x > self.HighLogThreshold):
            y = 0
        else:
            y = self.RBF(x)

        if self.FunctionType == 1:
            tau = 1.0/(y + 1.0/self.tau0 + t)
        elif self.FunctionType == 2:
            tau = 1.0/((y+1.0)*(1.0/self.tau0 + t))
        elif self.FunctionType == 3:
            tau = (1.0-y)/(1.0/self.tau0 + t)
        else:
            raise ValueError('Invalid function type.')

        Trace = self.trace_Binv*tau

        return Trace
