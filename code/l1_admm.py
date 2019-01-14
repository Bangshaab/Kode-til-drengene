import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time as t


def l1_admm(A,b,rho,maxit,x = True,z = True,u = True, er = 10**(-14), es = 10**(-14),quiet = False,datatype = np.float64):

    def proj(v, pA,x0):
        return np.dot(pA, v) + x0

    def soft(v, kappa):
        retval = (1 - kappa / np.absolute(v)) * v
        retval[np.absolute(v) < kappa] = 0
        return retval
    
    # Timing for update steps.
    x_time = 0
    z_time = 0
    hist_time = 0
    pre_time = 0
    st_global_sum = 0

    # Matrix Calculation
    st = t.time()
    mA = np.linalg.pinv(A).astype(datatype)
    pA = (np.eye(mA.shape[0]) - np.dot(mA, A)).astype(datatype)
    x0 = np.dot(mA, b).astype(datatype)

    if x:
        x = np.zeros([A.shape[1],1],dtype = datatype)
    if z:
        z = np.zeros([A.shape[1],1],dtype = datatype)
    if u:
        u = np.zeros([A.shape[1],1],dtype = datatype)
    
    historyx = []
    historyz = []
    historyu = []
    historyr = []
    historys = []

    r_norm = 1000
    s_norm = 1000
    i = 0
    pre_time = t.time()-st
    #Do iteration
    while (r_norm + s_norm > es + er) and i < maxit:
        #x-minimization-step
        st_global = t.time() #Global time
        st = t.time() # X-minimization time 
        x = proj(z-u,pA,x0)
        x_time += t.time()-st

        #z.minimization-step
        st = t.time()
        z_old = z.copy()
        z = soft(u+x,1/rho)
        #z = np.round(z,2)
        z_time += t.time()-st

        #u update-step
        u = u + x - z

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
        st_global_sum = st_global_sum + t.time()-st_global 
        if not(quiet):
            print("Iteration:",i,"Error:",r_norm+s_norm)
    if not(quiet):
        print("x_time:",x_time/i)
        print("z_time:", z_time / i)
        print("hist_time:", hist_time / i)
        print("Setup time:", pre_time)
    avg_it_time = st_global_sum/i
    avg_x_time = x_time/i
    avg_z_time = z_time/i
    return [historyx,historyz,historyu,historyr,historys,avg_it_time,i]


def l1_admm_nonoptimized(A,b,rho,maxit,x = True,z = True,u = True, er = 10**(-14), es = 10**(-14),quiet = False,datatype = np.float64):

    def proj(v, pA,x0):
        return np.dot(pA, v) + x0

    def soft(v, kappa):
        retval = (1 - kappa / np.absolute(v)) * v
        retval[np.absolute(v) < kappa] = 0
        return retval
    
    # Timing for update steps.
    x_time = 0
    z_time = 0
    hist_time = 0
    pre_time = 0
    st_global_sum = 0

    # Matrix Calculation
    st = t.time()
    if x:
        x = np.zeros([A.shape[1],1],dtype = datatype)
    if z:
        z = np.zeros([A.shape[1],1],dtype = datatype)
    if u:
        u = np.zeros([A.shape[1],1],dtype = datatype)
    
    historyx = []
    historyz = []
    historyu = []
    historyr = []
    historys = []

    r_norm = 1000
    s_norm = 1000
    i = 0
    pre_time = t.time()-st
    #Do iteration
    while (r_norm + s_norm > es + er) and i < maxit:
        #x-minimization-step
        st_global = t.time()
        st = t.time()
        x = proj(z-u,(np.eye(A.shape[1]) - np.dot(np.linalg.pinv(A).astype(datatype), A)).astype(datatype),np.dot(np.linalg.pinv(A).astype(datatype), b).astype(datatype))
        x_time += t.time()-st

        #z.minimization-step
        st = t.time()
        z_old = z.copy()
        z = soft(u+x,1/rho)
        #z = np.round(z,2)
        z_time += t.time()-st

        #u update-step
        u = u + x - z

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
        st_global_sum = st_global_sum + t.time()-st_global 
        if not(quiet):
            print("Iteration:",i,"Error:",r_norm+s_norm)
    if not(quiet):
        print("x_time:",x_time/i)
        print("z_time:", z_time / i)
        print("hist_time:", hist_time / i)
        print("Setup time:", pre_time)
    avg_it_time = st_global_sum/i
    return [historyx,historyz,historyu,historyr,historys,avg_it_time,i]


#Test
'''
dx = 1000
deq = 100
c = 1
noise = 0
A = np.random.rand(deq,dx)
b_a = (A[:,np.random.choice(dx,c)]).copy()
b = np.expand_dims(np.sum(b_a,1),1)
print(b.shape)
e = np.random.randn(deq,1)*noise
b = b + e

res = l1_admm_nonoptimized(A,b,0.01,1000)

plt.semilogy(np.array(res[3]))
plt.semilogy(np.array(res[4]))
plt.figure(2)
plt.semilogy(np.array(res[4])+np.array(res[3]))
#plt.stem(res[0][-1])


plt.show()
'''