import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../kode')
import alphaLasso as lasso

N = 200 # Number of variables
P = int(N/4) # No equations
rho = 1 # rho
lamda = 1 # lambda
A = np.random.rand(P,N) # Random A matrix
x0 = np.zeros(N) # Wanted x_vector

x0[1] = 1 # Wanted x vector
x0[50] = -1 # Wanted x vector
b = (A @ x0) # Creation of b
aaa = np.zeros(100)
bits = 16
res = lasso.lassoQuantPre(A,b,rho,lamda,100,aaa,xPre = True,uEst = True,zEst = True,err = -1,rounding = True,roundSpecs=[-1.5,1.5,int(bits)]) # Lasso
plt.stem(res[0][-1])
plt.xlabel('i')
plt.ylabel('$x_i$')
plt.annotate("Type 1", xy=(1, res[0][-1][1]), xytext=(11, res[0][-1][1]-0.05),arrowprops=dict(arrowstyle="->"))
plt.annotate("Type 2", xy=(50, res[0][-1][50]), xytext=(60, res[0][-1][50]+0.05),arrowprops=dict(arrowstyle="->"))
plt.annotate("Type 3", xy=(100, res[0][-1][100]), xytext=(110, res[0][-1][100]+0.1),arrowprops=dict(arrowstyle="->"))

#plt.savefig('arti.eps')

plt.figure(2)
plt.plot(np.array(res[0])[:,1],label = '$x_1^k$')
plt.plot(np.array(res[3])[:,1],label = '$u_1^k$')
plt.title('Realization of Type 1')
plt.xlabel('Iterations')
plt.ylabel('$x_1^k$')
plt.legend()
plt.savefig('diffx.eps')


plt.figure(3)
plt.plot(np.array(res[0])[:,50],label = '$x_{50}^k$')
plt.plot(np.array(res[3])[:,50],label = '$u_{50}^k$')
plt.title('Realization of Type 2')
plt.xlabel('Iterations')
plt.ylabel('$x_{50}^k$')
plt.legend()
plt.savefig('diffx-.eps')


plt.figure(4)
plt.plot(np.array(res[0])[:,100],label = '$x_{100}^k$')
plt.plot(np.array(res[3])[:,100],label = '$u_{100}^k$')
plt.title('Realization of Type 3')
plt.xlabel('Iterations')
plt.ylabel('$x_{100}^k$')
plt.legend()
plt.savefig('zero.eps')

plt.show()