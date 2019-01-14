import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time as t

def lasso(A,
          b,
          rho,
          lmbda,
          maxit,
          x = True,
          z = True,
          u = True,
          er=0,
          es = 0,
          quiet = True,
          datatype = np.float64,
          rounding = False,
          roundSpecs = [1,1,1]):

    def soft(v, kappa):
        retval = (1 - kappa / np.absolute(v)) * v
        retval[np.absolute(v) < kappa] = 0
        return retval
#Rounding
    if rounding == True:
        limits = np.linspace(roundSpecs[0],roundSpecs[1],2**roundSpecs[2])
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
#Timing
    x_time = 0
    z_time = 0
    hist_time = 0
    pre_time = 0
    st_global_sum = 0

#Matrix
    I = np.identity(A.shape[1])
    st = t.time()
    As = (np.linalg.inv((A.T.dot(A)+rho*I))*rho)
    bs = (np.linalg.inv(A.T.dot(A)+rho*I)).dot(A.T.dot(b))



    if type(x) == bool:
        x = np.zeros([A.shape[1]],dtype = datatype)
    if type(z) == bool:
        z = np.zeros([A.shape[1]],dtype = datatype)
    if type(u) == bool:
        u = np.zeros([A.shape[1]],dtype = datatype)

    #print(x[0])

    historyx = []
    historyz = []
    historyu = []
    historyr = []
    historys = []

    r_norm = 1000
    s_norm = 1000
    i = 0
    pre_time = t.time()-st

    while i < maxit:
        #x-minimization-step
        st_global = t.time()
        st = t.time()
        x = As.dot(z-u)+bs
        x_time += t.time()-st
        
        if rounding == True:
            x = rounderino(limits,a,x).copy()

        #z.minimization-step
        st = t.time()
        z_old = z.copy()
        z = soft(u+x,lmbda/rho)
        #z = np.round(z,2)
        z_time += t.time()-st

        if rounding == True:
            z = rounderino(limits,a,z).copy()

        #u update-step
        u = u+x-z

        if rounding == True:
            u = rounderino(limits,a,u).copy()

        #Error updates
        r = x - z
        s = rho*(z - z_old)
        r_norm = np.asscalar(np.linalg.norm(r))
        s_norm = np.asscalar(np.linalg.norm(s))

        #History update
        st = t.time()
        historyx.append(x.copy())
        historyz.append(z.copy())
        historyu.append(u.copy())
        historyr.append(r_norm)
        historys.append(s_norm)
        hist_time = t.time()-st
        i += 1
        st_global_sum = st_global_sum + t.time() - st_global
        if not (quiet):
            print("Iteration:", i, "Error:", r_norm + s_norm)
    if not (quiet):
        print("x_time:", x_time / i)
        print("z_time:", z_time / i)
        print("hist_time:", hist_time / i)
        print("Setup time:", pre_time)
    avg_it_time = st_global_sum / i
    avg_x_time = x_time / i
    avg_z_time = z_time / i
    return [historyx, historyz, historyu, historyr, historys, avg_it_time, i]


N = 1000
P = 250
np.random.seed(42)
A = np.random.randn(P,N)
b = np.sum(A[:,np.random.choice(N,1)].copy(),1)
rho = 1
lmbda = 1

lowerBound = -1
upperBound = 1
maxIt = 200
reps = 10
for l in range(4,16):
    print(l+1,'of 16')
    for h in range(reps):
        res = lasso(A,b,rho,lmbda,maxIt,rounding=True,roundSpecs=[lowerBound,upperBound,(l+1)])
        if h > 0:
            tempR = (np.divide(np.array(res[3]),reps) + tempR).copy()
            tempS = (np.divide(np.array(res[4]),reps) + tempS).copy()
        else:
            tempR = np.divide(np.array(res[3]),reps).copy()
            tempS = np.divide(np.array(res[4]),reps).copy()

    if l > 4:
        histR = np.append(histR,tempR,axis=0).copy()
        histS = np.append(histS,tempS,axis=0).copy()
    else:
        histR = tempR.copy()
        histS = tempS.copy()

print('17 of 16')    
for h in range(reps):    
    resNo = lasso(A,b,rho,lmbda,maxIt)
    if h > 0:
        tempR = (np.divide(np.array(res[3]),reps) + tempR).copy()
        tempS = (np.divide(np.array(res[4]),reps) + tempS).copy()
    else:
        tempR = np.divide(np.array(res[3]),reps).copy()
        tempS = np.divide(np.array(res[4]),reps).copy()

histR = np.append(histR,tempR,axis=0).copy()
histS = np.append(histS,tempS,axis=0).copy()

fig = plt.figure(figsize=(10.6,6))
legend = []

for l in range(13):
    if l < 10:
        plt.plot(np.linspace(1,maxIt,maxIt),histR[l*maxIt:(l+1)*maxIt]+histS[l*maxIt:(l+1)*maxIt])
    else:
        plt.plot(np.linspace(1,maxIt,maxIt),histR[l*maxIt:(l+1)*maxIt]+histS[l*maxIt:(l+1)*maxIt],'--')
    if l < 12:
        legend.append('%d bits precision'%(l+5))

plt.plot(np.linspace(1,maxIt,maxIt),np.linspace(0.2,0.2,maxIt),':',color='k')

#plt.plot(np.array(resNo[3])+np.array(resNo[4]))
legend.append('64 bits precision')
legend.append('Error of 0.2')
plt.xlabel('Iterations')
plt.ylabel('$e_{pri} + e_{dual}$')
plt.legend(legend)
plt.title('Convergence of lasso for different levels of quatization')

#fig.savefig('lassoConvergenceQuantz.eps')
plt.show()
