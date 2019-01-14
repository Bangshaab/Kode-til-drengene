import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time as t

def lass_admm(A,b,rho,lamda,maxit,x = True,z = True,u = True, er = 10**(-9), es = 10**(-9),quiet = False):
    
    def proj(v,x_const,Aa):
        return x_const + np.dot(Aa,v)

    def soft(v,kappa):
        retval = (1 - kappa / np.absolute(v)) * v
        retval[np.absolute(v) < kappa] = 0
        return retval
    
    x_time = 0
    z_time =0

    mA = np.linalg.inv(np.dot(A.T,A)+rho*np.eye(A.shape[1]))
    x_const = np.dot(mA,np.dot(A.T,b))
    mA = rho*mA

    if x:
        x = np.zeros([A.shape[1],1])
    if z:
        z = np.zeros([A.shape[1],1])
    if u:
        u = np.zeros([A.shape[1],1])

    historyx = []
    historyz = []
    historyu = []
    historyr = []
    historys = []

    #Do iteration
    r_norm = 1000
    s_norm = 1000
    i = 0
    while (r_norm+s_norm > er+es) and i < maxit:
        #x-minimization-step
        st = t.time()
        x = proj(z-u,x_const,mA)
        x_time += t.time()-st

        #z.minimization-step
        st = t.time()
        z_old = z.copy()
        z = soft(u+x,lamda/rho)
        z_time += t.time()-st

        #u update-step
        u = u + x - z

        #Error update
        r = x - z
        s = rho*(z - z_old)
        r_norm = np.asscalar(np.linalg.norm(r))
        s_norm = np.asscalar(np.linalg.norm(s))

        historyx.append(x.copy())
        historyz.append(z.copy())
        historyu.append(u.copy())
        historyr.append(r_norm)
        historys.append(s_norm)

        if not(quiet):
            print("Iteration:",i,"Error",r_norm)
        i += 1
    if not(quiet):
        print("x_time:",x_time/maxit)
        print("z_time:", z_time / maxit)
    return [historyx,historyz,historyu,historyr,historys]

# TEST
'''
dx = 10000
deq = 1000
noise = 0
noise_bit = 0
A = np.random.randn(deq,dx)
b = np.random.randn(deq,1)
e_choise = np.random.choice(deq, int(deq*noise_bit))
b = np.expand_dims(A[:,10].copy(),1)
e = np.random.randn(deq,1)*noise
b = b + e
bef = t.time()
res = lass_admm(A,b,0.5,10,1000)
print("Time taken:",t.time()-bef)
plt.plot(np.array(res[-2]))
plt.plot(np.array(res[-1]))
plt.figure(2)
plt.stem(res[0][-1])
plt.show()
'''