#! /usr/bin/env python

import faulthandler; faulthandler.enable()

import TraceInv

A = TraceInv.GenerateMatrix(NumPoints=10)
t1 = TraceInv.ComputeTraceOfInverse(A)
print(t1)
t2 = TraceInv.ComputeTraceOfInverse(A,ComputeMethod='SLQ')
print(t2)
