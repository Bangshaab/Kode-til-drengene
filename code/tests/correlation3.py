import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../kode')
import alphaLasso as lasso

bits = input('Enter numbers of bits: ')

N = 1000 # Number of variables
P = int(N/4) # No equations
rho = 1 # rho
lamda = 1 # lambda
A = np.random.rand(P,N) # Random A matrix
x0 = np.zeros(N) # Wanted x_vector
noise = np.random.randn(P)*0 # Noise to the sample
c = np.random.choice(N,1)
x0[c] = 1 # Wanted x vector
b = (A @ x0) + noise # Creation of b
sigma = 0.01
mu = 0
its = 100
data = []
data_ori = []
data_hat = []
#print(x0[c])
print('Running with',bits,'bits')
print('Running with',its,'iterations')
for i in range(its):

    A = np.random.rand(P,N) # Random A matrix
    c = np.random.choice(N,10)
    x0 = np.zeros(N) # Wanted x_vector
    x0[c] = np.random.randint(0,2,size = c.shape[0])*2-1 # Wanted x vector
    b = (A @ x0) + noise # Creation of b
    print('iteration',i)
    x_start = np.random.randn(N)*np.sqrt(sigma) + mu
    z_start = np.random.randn(N)*np.sqrt(sigma) + mu
    u_start = np.random.randn(N)*np.sqrt(sigma) + 0#lamda/rho
    #print(x_start)
    #aaa = np.load('zetas/zeta16.npy')
    aaa = np.zeros(100)*1
    res = lasso.lassoQuantPre(A,b,rho,lamda,100,aaa,xPre = x_start,uEst = u_start,zEst = z_start,err = -1,rounding = False,roundSpecs=[-1.5,1.5,int(bits)]) # Lasso
    X = np.array(res[-1]) # History of the x vector
    Xhat = np.array(res[1])

    for k in c:
        data_ori.append(X[:,k])
        data_hat.append(Xhat[:,k])
        '''
        if k < N-1:
            data_ori.append(X[:,k+1])
            data_hat.append(Xhat[:,k+1])
        '''

    #dX = X[1:,:]-X[:-1,:] # History of difference of the x vector
    #for k in c:
    #    data.append(X[:,k])
    #    if k < N-1:
    #        data.append(X[:,k+1])
    #plt.stem(X[-1,:])
    #plt.show()
    '''
    #print(np.array(res[0]).shape)
    c0 = np.correlate(dX[:,0],dX[:,0],'same') # Autocorelation af første entry i dX
    #print(np.mean(dX[:,0]))
    #plt.plot(c0)
    plt.figure(0)
    plt.plot(X[:,0]) # Første entry af X udvikling i tid
    plt.figure(i+1)
    plt.stem(X[-1,:]) # Endelig løsning af X
    plt.figure(i+100)
    plt.plot(dX[:,0]) # Første entrys udvikoing over tid af DX
    '''
np.save('data/pfg_0.npy',np.array(data_ori))
exit()
max_val = max((i.shape[0] for i in data))
for i in range(len(data)):
    np.append(data[i],[0]*(max_val-data[i].shape[0]))

#print([i.shape[0] for i in data])
f1 = plt.figure(1)
plt.grid()
f1.set_size_inches(9,4.5)
plt.subplot(1,2,1)
plt.grid(1,alpha = 0.5)
plt.xlabel('Iterations [k]')
plt.title('Realizations of $X_i$')
plt.ylabel('$x_i$')
for i in data[0:20]:
    plt.plot(i)
datam = np.array(data)
sigma_hat = np.var(datam,0)
plt.subplot(1,2,2)
plt.grid(1,alpha = 0.5)
plt.title('Estimated variance of $X_i$')
plt.xlabel('Iterations [k]')
plt.ylabel('$\sigma_{X_i}$')
plt.plot(sigma_hat)
plt.tight_layout()
#f1.savefig('x_var.eps')

plt.figure(3)
c = []
cc = []
for i in data:
    c.append(np.correlate(i,i,'full'))
    cc.append(np.corrcoef(i))
cray  = np.corrcoef(data)
print(cray)
corr = np.array(c)
print(corr.shape)
ccorr = np.array(c)
m_corr = np.mean(corr,0)[100:]
m_ccor = np.mean(corr,0)[100:]
corr_coef = m_corr/m_corr[0]
m_ccor = m_ccor/m_ccor[0]
plt.plot(corr_coef)
plt.plot(m_ccor)
plt.title('Estimated correlation coefficient of $X_i$')
plt.xlabel('Iterations [k]')
plt.ylabel('$rho_{k}$')
#plt.savefig('corr1.eps')

#np.save('ACP.npy',m_corr)
#np.save('CC.npy',m_ccor)
np.save('Xdata3.npy',data)

plt.show() # Show plots. 