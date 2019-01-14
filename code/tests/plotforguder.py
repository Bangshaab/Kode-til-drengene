import numpy as np
import matplotlib.pyplot as plt

X0 = np.absolute(np.load('data/pfg_0.npy'))
X1 = np.absolute(np.load('data/pfg_1.npy'))
X102 = np.absolute(np.load('data/pfg_102.npy'))
X098 = np.absolute(np.load('data/pfg_098.npy'))
Xzeta = np.absolute(np.load('data/pfg_zeta.npy'))

mean_X0 = np.mean(X0,0)
mean_X1 = np.mean(X1,0)
mean_X102 = np.mean(X102,0)
mean_X098 = np.mean(X098,0)
mean_Xzeta = np.mean(Xzeta,0)

sigmaX0 = np.std(X0,0)
sigmaX1 = np.std(X1,0)
sigmaX102 = np.std(X102,0)
sigmaX098 = np.std(X098,0)
sigmaXzeta = np.std(Xzeta,0)

t = np.arange(98)


plt.plot(mean_X0[2:])
plt.plot(mean_X1[2:])
#plt.plot(mean_X102)
#plt.plot(mean_X098)
plt.plot(mean_Xzeta[2:])
plt.fill_between(t,mean_X0[2:]-sigmaX0[2:]*2,mean_X0[2:]+sigmaX0[2:]*2,alpha=0.5, label='$\\alpha^k=0$')
plt.fill_between(t,mean_X1[2:]-sigmaX1[2:]*2,mean_X1[2:]+sigmaX1[2:]*2,alpha=0.5, label='$\\alpha^k=1$')
plt.fill_between(t,mean_Xzeta[2:]-sigmaXzeta[2:]*2,mean_Xzeta[2:]+sigmaXzeta[2:]*2,alpha=0.5, label='$\\alpha^k=\\zeta^k$')
#print(mean_X0.shape)
plt.grid(alpha = 0.5)
plt.xlim(0,50)
plt.xlabel('Iteration (k)')
plt.ylabel('E[|${\\tilde{x}}$|]')
plt.legend()
plt.savefig('etplotforguder.pdf')
plt.show()

'''
plt.figure(2)
plt.errorbar(t,mean_X0[2:],sigmaX0[2:]*2, color = 'green', capsize=3, label='$\\alpha^k=0$')
plt.errorbar(t,mean_X1[2:],sigmaX1[2:]*2, color = 'purple', capsize=3, label='$\\alpha^k=1$')
plt.errorbar(t,mean_Xzeta[2:],sigmaXzeta[2:]*2, color = 'orange', capsize=3, label='$\\alpha^k=\\zeta^k$')
plt.xlim(-1,50)
plt.xlabel('Iteration')
plt.ylabel('$\\tilde{x}$')
plt.legend()
plt.savefig('etplotforguder.eps')
plt.show()
'''