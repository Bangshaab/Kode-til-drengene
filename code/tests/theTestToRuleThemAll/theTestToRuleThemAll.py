import numpy as np
import matplotlib.pyplot as plt
import time as t

bitDepth = 5

def lassoQuantPre(A,b,rho,lmbda,maxit,alpha,x = True,z = True,u = True,err=10**(-9),rounding = False,roundSpecs = [1,1,1]):  
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
    

  if type(x) == bool:
      xPre = np.zeros([A.shape[1]])
  if type(z) == bool:
      zPre = np.zeros([A.shape[1]])
      zEst = np.zeros([A.shape[1]])
  if type(u) == bool:
      uPre = np.zeros([A.shape[1]])
      uEst = np.zeros([A.shape[1]])

    
  histXEst = []
  histZEst = []
  histUEst = []
  historyr = []
  historys = []


  r_norm = 1000
  s_norm = 1000
  i = 0

  while (r_norm + s_norm > err) and i < maxit:
    #x-minimization-step
    xEst = As.dot(zEst-uEst)+bs
    xErr = xEst - alpha[i]*xPre
    xQua = rounderino(limits,a,xErr)

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
    histXEst.append(xQua.copy())
    histZEst.append(zEst.copy())
    histUEst.append(uEst.copy())
        
        #historyx.append(xTrue.copy())
        #historyz.append(zTrue.copy())
        #historyu.append(uTrue.copy())
    historyr.append(r_norm)
    historys.append(s_norm)

    i += 1

  return [histXEst, histZEst, histUEst, historyr, historys, i]


reps = 1000
maxItt = 2000
N = 1000
P = 250

np.random.seed(42)
r = 0.3
R = np.zeros((reps,maxItt-1))
alpha = np.load('zeta16.npy')
solved = []
for i in range(reps):
    A = np.random.randn(P, N)
    c = np.random.choice(N, 1)
    b = np.sum(A[:,c].copy(), 1)+np.random.randn(P)
    xw = np.zeros(N)
    xw[c] = 1
    xw = xw.astype(int)


    res = lassoQuantPre(A,b,1,1,maxItt,alpha,err=-10000,rounding=True,roundSpecs=[-r,r,bitDepth])

    solved.append(np.allclose(xw,np.around(res[0]).copy().astype(int)))

    tempJ = np.asarray(res[3]) + np.asarray(res[4])
    R[i,:] = tempJ[1:].copy()

    if i%10==0:
        print('Bitdepth: %d   alpha = zeta: '%bitDepth,i,'of',reps)
np.save('R%d_a=Z_solved=%d.npy'%(bitDepth,sum(solved)),R)

np.random.seed(42)
r = 1.6
R = np.zeros((reps,maxItt-1))
alpha = np.zeros(reps)
solved = []
for i in range(reps):
    A = np.random.randn(P, N)
    c = np.random.choice(N, 1)
    b = np.sum(A[:,c].copy(), 1)+np.random.randn(P)
    xw = np.zeros(N)
    xw[c] = 1
    xw = xw.astype(int)


    res = lassoQuantPre(A,b,1,1,maxItt,alpha,err=-10000,rounding=True,roundSpecs=[-r,r,bitDepth])

    solved.append(np.allclose(xw,np.around(res[0]).copy().astype(int)))

    tempJ = np.asarray(res[3]) + np.asarray(res[4])
    R[i,:] = tempJ[1:].copy()

    if i%10==0:
        print('Bitdepth: %d   alpha = zeta: '%bitDepth,i,'of',reps)
np.save('R%d_a=0_solved=%d.npy'%(bitDepth,sum(solved)),R)

np.random.seed(42)
r = 0.5
R = np.zeros((reps,maxItt-1))
alpha = np.ones(reps)
solved = []
for i in range(reps):
    A = np.random.randn(P, N)
    c = np.random.choice(N, 1)
    b = np.sum(A[:,c].copy(), 1)+np.random.randn(P)
    xw = np.zeros(N)
    xw[c] = 1
    xw = xw.astype(int)


    res = lassoQuantPre(A,b,1,1,maxItt,alpha,err=-10000,rounding=True,roundSpecs=[-r,r,bitDepth])

    solved.append(np.allclose(xw,np.around(res[0]).copy().astype(int)))

    tempJ = np.asarray(res[3]) + np.asarray(res[4])
    R[i,:] = tempJ[1:].copy()

    if i%10==0:
        print('Bitdepth: %d   alpha = zeta: '%bitDepth,i,'of',reps)
np.save('R%d_a=1_solved=%d.npy'%(bitDepth,sum(solved)),R)
