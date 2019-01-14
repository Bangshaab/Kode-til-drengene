#Simple Lasso comparison
import numpy as np
import sys
import time as t
import matplotlib.pyplot as plt
sys.path.insert(0, '../kode')
import lass_admm as jakob
import Lasso as jonas

n = 10000
p = int(n/2)

A = np.random.rand(p,n)
x = np.zeros([n,1])
x[0,0] = 1
b = A.dot(x)

rho = 1
lamda = 5
maxit = 1000
er = 10**(-9)
es = 10**(-9)

st = t.time()
res_jonas = jonas.lasso(A,b,rho,lamda,maxit,er = er,es = es) 
print("Jonas' in:",t.time()-st)
st = t.time()
res_jakob = jakob.lass_admm(A,b,rho,lamda,maxit,er = er,es = es) 
print("Jakobs in:",t.time()-st)

plt.stem(res_jonas[0][-1])
plt.figure(2)
plt.stem(res_jakob[0][-1])
plt.show()