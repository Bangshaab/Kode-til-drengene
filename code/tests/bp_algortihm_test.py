import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../kode')
import l1_admm as l1
import Lasso as lasso

#n = 1000 # Number og variables
q = 10
lamda = 100
p = 100 # Number og equations
rho  = 100 # Chosen rho
maxit = 1000 # Maximum iteriations. Should not be relevant!
er = 1/2*10**(-9) # Errors
es = 1/2*10**(-9)
time_non = []
time_opt = []
it_non = []
it_opt = []

logspace = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
for n in logspace:
    print("n:",n)
    time_opt_temp = 0
    time_non_temp = 0
    it_opt_temp = 0
    it_non_temp = 0
    for i in range(q):
        A = np.random.rand(int(n/2),n) # Creation of random matrix
        x = np.zeros([n,1]) # Creation og wanted solution x
        x[0,0] = 1
        b = A.dot(x) # Resulting vector b

        res_optimal = l1.l1_admm(A,b,rho,maxit,er =er,es = es,quiet = True)
        res_nonoptimal = l1.l1_admm_nonoptimized(A,b,rho,maxit,er =er,es = es,quiet = True)
        time_opt_temp += res_optimal[5]
        time_non_temp += res_nonoptimal[5]
        it_opt_temp += res_optimal[6]
        it_non_temp += res_nonoptimal[6]

    time_opt.append(time_opt_temp/q)
    it_opt.append(it_opt_temp/q)
    time_non.append(time_non_temp/q)
    it_non.append(it_non_temp/q)
    count = len(time_opt)
    data = np.array([logspace[:count],time_non,time_opt,it_non,it_opt])
    np.save('data/Avg_iteration_time.npy',data)

fig, ax1 = plt.subplots()
ax1.semilogx(logspace,time_opt,'b')
#ax1.semilogx(logspace,time_non,'--b')
ax2 = ax1.twinx()
ax2.semilogx(logspace,it_opt,'r')
#ax2.semilogx(logspace,it_non,'--r')
ax2.set_ylabel('Iteratins [\cdot]')
ax1.set_ylabel('Average ieration time [s]')
plt.title('Average iteration time')

data = np.array([logspace,time_non,time_opt,it_non,it_opt])
np.save('data/Avg_iteration_time.npy',data)
print(data)

plt.show()


