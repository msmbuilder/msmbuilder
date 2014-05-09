"""
Implementation of a basic interior point algorithm.

@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""
# Author: Bharath Ramsundar <bharath.ramsundar@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import pdb

class InteriorPointSolver(object):
    """
    Implementation of a basic log-barrier interior point method for
    optimization problem:
        max f(X)
        subject to
            g_i(X) <= 0
            Tr(A X) = b

    We introduce logarithmic barrier

        phi(X) = -log(-g_i(X))

    We then iteratively solve the central path solution x*(t) of the
    problem below

        min t f(X) + phi(X)
        subject to
            Ax = b

    through constrained Newton optimization of the KKT conditions
    """
    def __init__(self, f, grad_f, hessian_f,
            gs, grad_gs, hessian_gs, A, b):
        """
        f: function
            Objective to maximize
        grad_f: function
            Gradient of objective
        hessian_f: function
            Hessian of objective
        gs: list
            list of constraint functions g_i
        grad_gs: list
            list of functions grad_g_i, where grad_g_i computes
            the gradient of g_i
        hessian_gs: list
            list of functions hessian_g_i, where hessian_g_i computes
            the hessian of g_i
        A: numpy.ndarray
            numpy matrix
        b: numpy.ndarray
            numpy vector. Constraint Tr(A X) = b
        """

        self.dim = dim
        self.f = f
        self.grad_f = grad_f
        self.hessian_f = hessian_f
        self.gs = gs
        self.grad_gs = grad_gs
        self.hessian_gs = hessian_gs
        self.A = A
        self.b = b

def barrier_method(eps, mu, f, grad_f, hessian_f,
        phi, grad_phi, hessian_phi, A, b):
    """
    eps: float
        errror tolerance
    mu: float
        mu > 0 provides the scaling per outer step of the barrier
        method.
    f: function
        Optimization objective
    grad_f: function
        Gradient of objective
    hessian_f: function
        Hessian of objective
    phi: function
        Barrier method
    grad_phi: function
        Gradient of Barrier
    hessian_phi: function
        Hessian of Barrier
    A: numpy.ndarray
        Constraint matrix
    b: numpy.ndarray
        Constraint Vector
    """
    X = None
    while True:
        def g(X):
            return t*f(X) +
        X = newton_method_with_equality(X, eps, t)
        if 1./t < eps:
            break
        t = mu * t
    return X

def feasible_start_newton_method(X, eps, g, grad_g, hessian_g, A, b):
    """
    X: numpy.ndarray
        feasible starting point
    eps: float
        error tolerance, eps > 0
    g: function
        function to optimize
    grad_g: function
        Gradient of objective
    hessian_g: function
        Hessian of objective
    A: numpy.ndarray
        constraint matrix
    b: numpy.ndarray
        Tr(A X) = b
    """
    while True:
        delta_xnt = compute_newton_step()
        lambda_x = compute_newton_decrement()
        if lambda_x**2/2. < eps:
            break
        t = backtracking_line_search(X, delta_xnt, lambda_x)
        X = X + t * delta_xnt
    return X

def infeasible_start_newton_method(eps, g, grad_g, hessian_g, A, b):
    """
    X: numpy.ndarray
        feasible starting point
    eps: float
        error tolerance, eps > 0
    g: function
        function to optimize
    grad_g: function
        Gradient of objective
    hessian_g: function
        Hessian of objective
    A: numpy.ndarray
        constraint matrix
    b: numpy.ndarray
        Tr(A X) = b
    """
    pass

def f():
    pass

dim = 3
gs = []
grad_gs = []
hessian_gs = []
A = np.random.rand(dim, dim)
b = np.random.rand(dim)
i = InteriorPointSolver(f, gs, grad_gs, hessian_gs, A, b)
