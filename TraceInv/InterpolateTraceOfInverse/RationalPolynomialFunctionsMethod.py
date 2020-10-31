# =======
# Imports
# =======

from __future__ import print_function
from .InterpolantBaseClass import InterpolantBaseClass

import numpy

# ====================================
# Rational Polynomial Functions Method
# ====================================

class RationalPolynomialFunctionsMethod(InterpolantBaseClass):

    # ----
    # Init
    # ----

    def __init__(self,A,InterpolantPoints,**Options):

        # Base class constructor
        super(RationalPolynomialFunctionsMethod,self).__init__(A,InterpolantPoints)

        # Initilaize interpolator
        self.Numerator = None
        self.Denominator = None
        self.InitializeInterpolator()

    # ----------------
    # Rational Poly 12
    # ----------------

    @staticmethod
    def RationalPoly12(eta_i,tau_i,tau0):
        """
        Rational polynomial of order 1 over 2
        """

        # Matrix of coefficients
        A = numpy.array([
            [eta_i[0],1-tau0/tau_i[0]],
            [eta_i[1],1-tau0/tau_i[1]]])

        # Vector of right hand side
        c = numpy.array([
            eta_i[0]/tau_i[0]-eta_i[0]**2,
            eta_i[1]/tau_i[1]-eta_i[1]**2])

        # Condition number
        print('Condition number: %0.2e'%(numpy.linalg.cond(A)))

        # Solve with least square. NOTE: do not solve with numpy.linalg.solve directly.
        b = numpy.linalg.solve(A,c)
        b0 = b[1]
        b1 = b[0]
        a0 = b0*tau0

        # Output
        Numerator = [1,a0]
        Denominator = [1,b1,b0]

        # Check poles
        Poles = numpy.roots(Denominator)
        if numpy.any(Poles > 0):
            print('Denominator poles:')
            print(Poles)
            raise ValueError('RationalPolynomial has positive poles.')

        return Numerator,Denominator

    # ----------------
    # Rational Poly 23
    # ----------------

    @staticmethod
    def RationalPoly23(eta_i,tau_i,tau0):
        """
        Rational polynomial of order 2 over 3
        """

        # Matrix of coefficients
        A = numpy.array([
            [eta_i[0]**2,eta_i[0],1-tau0/tau_i[0],-eta_i[0]/tau_i[0]],
            [eta_i[1]**2,eta_i[1],1-tau0/tau_i[1],-eta_i[1]/tau_i[1]],
            [eta_i[2]**2,eta_i[2],1-tau0/tau_i[2],-eta_i[2]/tau_i[2]],
            [eta_i[3]**2,eta_i[3],1-tau0/tau_i[3],-eta_i[3]/tau_i[3]]])

        # Vector of right hand side
        c = numpy.array([
            eta_i[0]**2/tau_i[0]-eta_i[0]**3,
            eta_i[1]**2/tau_i[1]-eta_i[1]**3,
            eta_i[2]**2/tau_i[2]-eta_i[2]**3,
            eta_i[3]**2/tau_i[3]-eta_i[3]**3])

        # Condition number
        print('Condition number: %0.2e'%(numpy.linalg.cond(A)))

        # Solve with least square. NOTE: do not solve with numpy.linalg.solve directly.
        b = numpy.linalg.solve(A,c)
        b2 = b[0]
        b1 = b[1]
        b0 = b[2]
        a1 = b[3]
        a0 = b0*tau0

        # Output
        Numerator = [1,a1,a0]
        Denominator = [1,b2,b1,b0]

        # Check poles
        Poles = numpy.roots(Denominator)
        if numpy.any(Poles > 0):
            print('Denominator poles:')
            print(Poles)
            raise ValueError('RationalPolynomial has positive poles.')

        return Numerator,Denominator

    # -------------
    # Rational Poly
    # -------------

    @staticmethod
    def RationalPoly(t,Numerator,Denominator):
        """
        Evaluates rationa polynomial
        """

        return numpy.polyval(Numerator,t) / numpy.polyval(Denominator,t)

    # -----------------------
    # Initialize Interpolator
    # -----------------------

    def InitializeInterpolator(self):
        """
        """
        
        print('Initialize interpolator ...')
        

        tau0 = self.T0 / self.n
        tau_i = self.trace_eta_i / self.n

        # Coefficients of a linear system
        if self.p == 2:
            self.Numerator,self.Denominator = RationalPolynomialFunctionsMethod.RationalPoly12(self.eta_i,tau_i,tau0)

        elif self.p == 4:
            self.Numerator,self.Denominator = RationalPolynomialFunctionsMethod.RationalPoly23(self.eta_i,tau_i,tau0)

        else:
            raise ValueError('In RationalPolynomial method, the number of interpolant points, p, should be 2 or 4.')

        print('Done.')

    # -----------
    # Interpolate
    # -----------

    def Interpolate(self,t):
        """
        Interpolates the trace of inverse of ``A + t*I``.

        :param t: A real variable to form the linear matrix function ``A + tI``.
        :type t: float

        :return: The interpolated value of the trace of inverse of ``A + tI``.
        :rtype: float
        """

        tau = RationalPolynomialFunctionsMethod.RationalPoly(t,self.Numerator,self.Denominator)
        T = tau*self.n

        return T
