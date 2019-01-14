import numpy as np
import matplotlib.pyplot as plt
import time as t
import subprocess


def lassoQuantPre(A,b,rho,lmbda,maxit,alpha,xPre = True,zEst = True,uEst = True,err=10**(-9),rounding = False,roundSpecs = [1,1,1]):  
  def soft(v, kappa):
    retval = (1 - kappa / np.absolute(v)) * v
    retval[np.absolute(v) < kappa] = 0
    return retval
  
  #Rounding
  if rounding == True:
    limits = -1*np.flip(np.linspace(roundSpecs[0],roundSpecs[1],2**roundSpecs[2],endpoint=False))
    a = np.convolve(limits,np.array([0.5,0.5]))[1:-1]

  def rounderino(limits,a,v):
      temp = np.searchsorted(a,v,side ='left')
      t = []
      for i in range(temp.size):
          if temp[i] == limits.size:
              t.append(limits[temp[i-1]])
          else:
              t.append(limits[temp[i]])
      return np.asarray(t)

    #Matrix
  I = np.identity(A.shape[1])
  st = t.time()
  As = (np.linalg.inv((A.T.dot(A)+rho*I))*rho)
  bs = (np.linalg.inv(A.T.dot(A)+rho*I)).dot(A.T.dot(b))


  if alpha.shape[0] <= maxit:
      alpha = np.append(alpha,np.ones([maxit-alpha.shape[0],1]))
    

  if type(xPre) == bool:
      xPre = np.zeros([A.shape[1]])
  if type(zEst) == bool:
      zPre = np.zeros([A.shape[1]])
      zEst = np.zeros([A.shape[1]])
  if type(uEst) == bool:
      uPre = np.zeros([A.shape[1]])
      uEst = np.zeros([A.shape[1]])

    
  histX = []
  histXHat = []
  histZEst = []
  histUEst = []
  historyr = []
  historys = []
  hist_qua = []

  histXHat.append(xPre.copy())  
  r_norm = 1000
  s_norm = 1000
  i = 0

  while (r_norm + s_norm > err) and i < maxit:
    #x-minimization-step
    xEst = As.dot(zEst-uEst)+bs
    xErr = xEst - alpha[i]*xPre
    
    if rounderino == True:
      xQua = rounderino(limits,a,xErr)
    else:
      xQua = xErr

    xPre = alpha[i]*xPre + xQua
            


        #z.minimization-step

    zEstOld = zEst.copy()
    zEst = soft(uEst+xPre,lmbda/rho)
    uEst = uEst + xPre - zEst

        
        #    z = soft(u+x,lmbda/rho)
        #    u = u+x-z

        #Error updates
    r = xEst - zEst
    s = rho*(zEst - zEstOld)
    r_norm = np.asscalar(np.linalg.norm(r))
    s_norm = np.asscalar(np.linalg.norm(s))

      #History update
    histX.append(xEst.copy())
    
    histZEst.append(zEst.copy())
    histUEst.append(uEst.copy())
    histXHat.append(xPre.copy())
    hist_qua.append(xQua.copy())  
        #historyx.append(xTrue.copy())
        #historyz.append(zTrue.copy())
        #historyu.append(uTrue.copy())
    historyr.append(r_norm)
    historys.append(s_norm)

    i += 1

  return [histX, histXHat, histZEst, histUEst, historyr, historys, i,hist_qua]

"""
reps = 1000
maxItt = 2000
N = 1000
P = 250
bitDepth = 8

np.random.seed(42)
r1 = .5
R = np.zeros((reps,maxItt-1))
alpha = np.ones(maxItt)


for i in range(reps):
    A = np.random.randn(P, N)
    c = np.random.choice(N, 1)
    b = np.sum(A[:,c].copy(), 1)+np.random.randn(P)

    res = lassoQuantPre(A,b,1,1,maxItt,alpha,err=-10000,rounding=True,roundSpecs=[-r1,r1,bitDepth])
    temp1 = np.asarray(res[3]) + np.asarray(res[4])

    R[i,:] = temp1[1:].copy()
    if i%10==0:
        print('alpha = 1: ',i,'of',reps)
np.save('R%d_a1.npy'%bitDepth,R)
"""