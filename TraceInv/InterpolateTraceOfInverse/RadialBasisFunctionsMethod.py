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

    # ----
    # Init
    # ----

    def __init__(self,A,InterpolantPoints,FunctionType=1):

        # Base class constructor
        super(RadialBasisFunctionsMethod,self).__init__(A,InterpolantPoints)

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
        """
        
        print('Initialize interpolator ...')
        
        # Take logarithm of eta_i
        xi = numpy.log10(self.eta_i)
        tau_i = self.trace_eta_i / self.n
        tau_0 = self.T0 / self.n

        if xi.size > 1:
            dxi = numpy.mean(numpy.diff(xi))
        else:
            dxi = 1

        # Function Type
        if self.FunctionType == 1:
            # Ascending function
            yi = 1/tau_i - (1/tau_0 + self.eta_i)
        elif self.FunctionType == 2:
            # Bell shape, going to zero at boundaries
            yi = (1/tau_i)/(1/tau_0 + self.eta_i) - 1
        elif self.FunctionType == 3:
            # Bell shape, going to zero at boundaries
            yi = 1 - (tau_i)*(1/tau_0 + self.eta_i)
        else:
            raise ValueError('Invalid function type.')

        # extend boundaries to zero
        self.LowLogThreshold = -4.5   # SETTING
        self.HighLogThreshold = 3.5   # SETTING
        NumExtend = 3            # SETTING
       
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
            self.RBF = scipy.interpolate.CubicSpline(xi,yi,bc_type=((1,0.0),(2,0)),extrapolate=True)       # best for ascneing function
            # self.RBF = scipy.interpolate.PchipInterpolator(xi,yi,extrapolate=True)                       # good
            # self.RBF = scipy.interpolate.UnivariateSpline(xi,yi,k=3,s=0.0)                               # bad
        elif (self.FunctionType == 2) or (self.FunctionType == 3):
            # These interpolation methods are good for the Bell shaped function
            self.RBF = scipy.interpolate.Rbf(xi,yi,function='gaussian',epsilon=dxi)                    # Best for function type 2,3,4
            # self.RBF = scipy.interpolate.Rbf(xi,yi,function='inverse',epsilon=dxi)
            # self.RBF = scipy.interpolate.CubicSpline(xi,yi,bc_type=((1,0.0),(1,0.0)),extrapolate=True)

        # Plot interpolation with RBF
        PlotFlag = False
        if PlotFlag:
            eta = numpy.logspace(self.LowLogThreshold-dxi,self.HighLogThreshold+dxi,100)
            x = numpy.log10(eta)
            y = self.RBF(x)
            fig,ax = plt.subplots()
            ax.plot(x,y)
            ax.plot(xi,yi,'o')
            ax.grid(True)
            ax.set_xlim([self.LowLogThreshold-dxi,self.HighLogThreshold+dxi])
            # ax.set_ylim(-0.01,0.18)
            plt.show()

        print('Done.')

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t):
        """
        Interpolates the trace of inverse of ``K + t*I``.

        :param t: A real variable to form the linear matrix function ``K + tI``.
        :type t: float

        :return: The interpolated value of the trace of inverse of ``K + tI``.
        :rtype: float
        """
        
        x = numpy.log10(t)
        if (x < self.LowLogThreshold) or (x > self.HighLogThreshold):
            y = 0
        else:
            y = self.RBF(x)

        tau_0 = self.T0 / self.n

        if self.FunctionType == 1:
            tau = 1/(y + 1/tau_0 + t)
        elif self.FunctionType == 2:
            tau = 1/((y+1)*(1/tau_0 + t))
        elif self.FunctionType == 3:
            tau = (1-y)/(1/tau_0 + t)
        else:
            raise ValueError('Invalid function type.')

        T = self.n*tau
        
        return T
