import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

#  ode
# First we write out the equation as a python functions
# dv/dt -av^2 + b = 0 -> write as dv/dt = f(t,v) this is what the solver is going to output
# Model dv/dt as a eqn with some inital a and b values for now, later we can add them in
# now we re factor: dv/dt = av^2 -b


def dvdt(t, v):
    return 3*v**2 - 5


v0 = 0

# now we can use the solver
# first we create all the time intervals we want to have
t = np.linspace(0, 1, 100)
# then we add our equation with the time intervals into the solver
# This first solver acts as a black box ensure that the first terme is your independant variable ie f(t,y) t is independant
# Returns each value of y for each timestep t as an array
sol_m1 = odeint(dvdt, y0=v0, t=t, tfirst=True)


# this second solver acts as a custamizable version of above where you can pick the method of solution
# returns a
sol_m2 = solve_ivp(dvdt, t_span=(0, max(t)), y0=[v0], t_eval=t)
# Note for this method you have to extract each solution individualy for each variable
# print(sol_m1)
# # solutions using sol_m2 need to be extracted
# print(sol_m2.t)
# print(sol_m2.y)

# For second order we need to use coupled ODE's
# heres the coupled example
# Coupled first order ODEs

# Consider the coupled first order ODEs:
# y1' = y1 + y2^2 + 3x,    y1(0) = 0
# y2' = 3y1 + y2^3 - cos(x),    y2(0) = 0

# Letting S = [y1, y2], we need to write a function that returns dS/dx = [dy1/dx, dy2/dx].
# The function dS/dx can take in S = [y1, y2] and x. This is like before, but in vector format.

# Vector representation:
# S = [y1, y2]^T
# dS/dx = f(x, S) = f(x, y1, y2) = [y1', y2']^T
#        = [y1 + y2^2 + 3x, 3y1 + y2^3 - cos(x)]^T


def dSdx1(x, S):
    # Each list index is one equation
    y1, y2 = S  # initalizes s as a vector function
    return [y1 + y2**2 + 3*x,
            3*y1 + y2**3 - np.cos(x)]


y1_0 = 0
y2_0 = 0
S_0 = (y1_0, y2_0)
x = np.linspace(0, 1, 100)
sol = odeint(dSdx1, y0=S_0, t=x, tfirst=True)
# In this case we will get a dimensional list with each  representing one variable
# Transpose so we can see the results as one list
y1_sol = sol.T[0]
y2_sol = sol.T[1]
# print(sol)
# print(y1_sol)
# print(y2_sol)
# with the other method
sol_m2 = solve_ivp(dSdx1, t_span=(0, max(x)), y0=[y1_0, y2_0], t_eval=x)
# print(sol_m2)
# with this second method finding each variable is fairly straight forward
y1_m2 = sol_m2.y[0]
y2_m2 = sol_m2.y[1]
print(y1_m2)
print(y1_sol)

# check weather both give the same result
assert y1_sol.sort() == y1_m2.sort(), "The two methods give diffrent results"

# For 2D ODE's we can simply represnt them as coupled ode's example
# Second Order ODEs

# Python does not have functions to directly solve second order ODEs.
# But any second order ODE can be converted into two first order ODEs.

# Consider the second order ODE:
# x'' = -x^2 + sin(x)

# We can convert this into two first order ODEs as follows:
# 1. Take x (this is what we're trying to solve for). Then define x' = v so that v becomes a new variable.
# 2. Note that x' = v is one differential equation.
# 3. Since v' = x'' = -x^2 + sin(x) = -v^2 + sin(x), we get another differential equation.

# Our two equations are:
# x' = v
# v' = -v^2 + sin(x)

# These are two coupled first order equations. They require an initial condition (x0 and v0).


def dSdx(x, S):
    x, v = S  # This line is the more important one it models what variables we want to solve using the S vector
    return [v,
            -v**2 + np.sin(x)]


x_0 = 0
v_0 = 5
S_0 = (x_0, v_0)
t = np.linspace(0, 1, 100)
sol = odeint(dSdx, y0=S_0, t=t, tfirst=True)
x_sol = sol.T[0]
v_sol = sol.T[1]

# mthod 2:
# for this function "y" is the solution
sol_m2 = solve_ivp(dSdx, t_span=(0, max(t)), y0=[x_0, y2_0], t_eval=x)
x_sol_2 = sol_m2.y[0]
v_sol_2 = sol_m2.y[1]
assert x_sol.sort() == x_sol_2.sort(), "The two methods give diffrent results"
