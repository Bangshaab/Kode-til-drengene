import numpy as np
import matplotlib.pyplot as plt
import time as t


def lasso(A,
          b,
          rho,
          lmbda,
          maxit,
          x=True,
          z=True,
          u=True,
          er=0,
          es=0,
          quiet=True,
          datatype=np.float64,
          rounding=False,
          roundSpecs=[1, 1, 1]):
    def soft(v, kappa):
        retval = (1 - kappa / np.absolute(v)) * v
        retval[np.absolute(v) < kappa] = 0
        return retval

    # Rounding
    if rounding == True:
        limits = np.linspace(roundSpecs[0], roundSpecs[1], 2 ** roundSpecs[2])
        a = np.convolve(limits, np.array([0.5, 0.5]))[1:-1]

    def rounderino(limits, a, v):
        temp = np.searchsorted(a, v, side='left')
        t = []
        for i in range(temp.size):
            if temp[i] == limits.size:
                t.append(limits[temp[i - 1]])
            else:
                t.append(limits[temp[i]])
        return np.asarray(t)

    # Timing
    x_time = 0
    z_time = 0
    hist_time = 0
    pre_time = 0
    st_global_sum = 0

    # Matrix
    I = np.identity(A.shape[1])
    st = t.time()
    As = (np.linalg.inv((A.T.dot(A) + rho * I)) * rho)
    bs = (np.linalg.inv(A.T.dot(A) + rho * I)).dot(A.T.dot(b))

    if type(x) == bool:
        x = np.zeros([A.shape[1]], dtype=datatype)
    if type(z) == bool:
        z = np.zeros([A.shape[1]], dtype=datatype)
    if type(u) == bool:
        u = np.zeros([A.shape[1]], dtype=datatype)

    # print(x[0])

    historyx = []
    historyz = []
    historyu = []
    historyr = []
    historys = []

    r_norm = 1000
    s_norm = 1000
    i = 0
    pre_time = t.time() - st

    while (r_norm + s_norm > .6) and i < maxit:
        # x-minimization-step
        st_global = t.time()
        st = t.time()
        x = As.dot(z - u) + bs
        x_time += t.time() - st

        if rounding == True:
            x = rounderino(limits, a, x)

        # z.minimization-step
        st = t.time()
        z_old = z.copy()
        z = soft(u + x, lmbda / rho)
        # z = np.round(z,2)
        z_time += t.time() - st

        if rounding == True:
            z = rounderino(limits, a, z)

        # u update-step
        u = u + x - z

        if rounding == True:
            u = rounderino(limits, a, u)

        # Error updates
        r = x - z
        s = rho * (z - z_old)
        r_norm = np.asscalar(np.linalg.norm(r))
        s_norm = np.asscalar(np.linalg.norm(s))

        # History update
        st = t.time()
        historyx.append(x.copy())
        historyz.append(z.copy())
        historyu.append(u.copy())
        historyr.append(r_norm)
        historys.append(s_norm)
        hist_time = t.time() - st
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
    return [historyx, historyz, historyu, historyr, historys, avg_it_time, i,x.copy()]


N = 1000
P = 250
#np.random.seed(42)

rho = 0.5
lmbda = 0.5

lowerBound = -1
upperBound = 1

bits = []
iteration = []
solved = []
meanItt = np.zeros(16)

for l in range(16):
    for h in range(20):
        A = np.random.randn(P, N)
        c = np.random.choice(N, 1)
        b = np.sum(A[:,c].copy(), 1)
        xw = np.zeros(N)
        xw[c] = 1
        xw = xw.astype(int)

        res = lasso(A,b, rho, lmbda, 5000, rounding=True, roundSpecs=[lowerBound, upperBound, l + 1])
        
        solved.append(np.allclose(xw,np.around(res[7]).copy().astype(int)))
        
        if l > 0:
            histX = np.append(histX, np.array(res[0]), axis=0)
            histR = np.append(histR, np.array(res[3]), axis=0)
            itt = np.append(itt, np.array(res[6]))
            iteration.append(res[6])
            bits.append(l+1)
        else:
            histX = np.array(res[0])
            histR = np.array(res[3])
            itt = np.array(res[6])
            iteration.append(res[6])
            bits.append(1)

    
    print('Bits: %d'%(l+1))
    print(solved)
    solved = []
        

iteration = np.array(iteration)
bits = np.array(bits)


hej = np.reshape(iteration,(l+1,h+1))
print(hej)
print(bits)

meanItt = np.mean(hej,axis=1)
fig = plt.figure()
plt.scatter(bits,hej)
print(meanItt)
mn = np.linspace(1,l+1,l+1)
print(mn)
plt.plot(mn,meanItt,'r')
plt.xlabel('Bits')
plt.ylabel('Iterations')
plt.grid()
fig.savefig('LassoQuantER', format='eps', dpi=1000)
plt.show()
#np.save('data/histX.npy', histX)
#np.save('data/histR.npy', histR)
#np.save('data/itt.npy', itt)

