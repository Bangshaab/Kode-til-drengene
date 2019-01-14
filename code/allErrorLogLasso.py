import numpy as np
import matplotlib.pyplot as plt
import time as t
from Lasso import lasso

bitDepth = 16
reps = 1000
maxItt = 5000
N = 1000
P = 250

np.random.seed(42)
"""
def lasso(A,
          b,
          rho,
          lmbda,
          maxit,
          x = True,
          z = True,
          u = True,
          er=10**(-9),
          es = 10**(-14),
          quiet = True,
          datatype = np.float64,
          rounding = False,
          roundSpecs = [1,1,1]):

return [historyx, historyz, historyu, historyr, historys, avg_it_time, i]
"""
R = np.zeros((reps,maxItt-1))

for i in range(reps):
    A = np.random.randn(P, N)
    c = np.random.choice(N, 1)
    b = np.sum(A[:,c].copy(), 1)+np.random.randn(P)

    res = lasso(A,b,1,1,maxItt,er=-10000,es=-10000,rounding=True,roundSpecs=[-1,1,bitDepth])
    temp = np.asarray(res[3]) + np.asarray(res[4])
    R[i,:] = temp[1:]
    if i%10==0:
        print(i,'of',reps)

np.save('R%d.npy'%bitDepth,R)