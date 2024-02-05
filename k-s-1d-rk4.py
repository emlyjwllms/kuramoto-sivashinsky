# solve u_t + u u_x - nu u_xx = 0 with 2nd order forward FD in space and RK4 in time
# nu = 0.1

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['font.size'] = 16
rcParams["figure.figsize"] = (5,8)
rcParams['figure.dpi']= 200
rcParams["figure.autolayout"] = True


# physical parameters
beta = 0.085

# spatial mesh
dx = 0.1
x0 = 0
xf = 10
x = np.arange(x0,xf+dx,dx)
nx = len(x)

# temporal mesh
dt = 0.000001
t0 = 0
tf = 20
t = np.arange(t0,tf+dt,dt)
nt = len(t)

# initial condition - random between -0.5 and 0.5
u0 = np.random.rand(nx-4) - 0.5

# solution structures
u_approx = np.zeros((nx,nt))
u_approx[2:-2,0] = u0

X, T = np.meshgrid(x,t)

def f(u,t):
    A = (-beta)*np.eye(nx-4,k=2) + (-0.5*u*dx**3 + 4*beta - dx**2)*np.eye(nx-4,k=1) + (2*dx**2 - 6*beta)*np.eye(nx-4) + (4*beta - dx**2 + 0.5*u*dx**3)*np.eye(nx-4,k=-1) + (-beta)*np.eye(nx-4,k=-2)
    return np.matmul(A,u)/dx**4

# time integration
for n in range(0,nt-1):

    #print(str(n) + " out of " +  str(nt-1) + ": " + str(n*100/(nt-1)) + "% complete")
    # RK4 time-stepping
    k1 = f(u_approx[2:-2,n], t[n])
    k2 = f(u_approx[2:-2,n] + k1*dt/2, t[n] + dt/2)
    k3 = f(u_approx[2:-2,n] + k2*dt/2, t[n] + dt/2)
    k4 = f(u_approx[2:-2,n] + k3*dt, t[n] + dt)
    u_approx[2:-2,n+1] = u_approx[2:-2,n] + (k1 + 2*k2 + 2*k3 + k4)*dt/6

    # homogeneous dirichlet BCs
    u_approx[0,n+1] = 0
    u_approx[-1,n+1] = 0

    # homogeneous neumann BCs
    u_approx[1,n+1] = 0
    u_approx[-2,n+1] = 0

np.save("u_approx",u_approx)
plt.contourf(X,T,u_approx.T)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.grid(True)
plt.title('Kuramoto-Sivashinsky')
plt.savefig("ks-rk4.png",dpi=300,format='png',transparent=True)
plt.show()