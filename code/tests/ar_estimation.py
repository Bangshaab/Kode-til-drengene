#AR(q) Estimation
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import sys
sys.path.insert(0, '../kode')
import Lasso as lasso

acf = np.load('ACP.npy')
coer_coef = np.load('CC.npy')
data = np.load('Xdata.npy')
acf = acf/acf[0]
order = 10

R = scipy.linalg.toeplitz(acf[:order])
r = acf[1:order+1]
phi = np.dot(np.linalg.inv(R),r)
col = 3
#print(R,r)
#print(phi)
phi_0 = coer_coef[0:order]
#Filtering
pred = np.zeros(data.shape[1]+1)
for i in range(order,data.shape[1]):
    x = data[col,(i-order):i]
    pred[i+1] = np.dot(phi,x)

dif = []

for k in range(data.shape[0]):
    pred_0 = np.zeros(data.shape[1])
    for i in range(order,data.shape[1]-1):
        x = data[k,(i-order):i]
        pred_0[i+1] = np.dot(phi_0,x[::-1])/order
    dif.append(pred_0-data[k,:])
#plt.plot(coer_coef)
dif = np.array(dif)
m_dif = np.mean(dif,0)
plt.figure(1)
plt.plot(m_dif)
plt.plot([order,order],[-0.1,0.1])
plt.figure(2)
plt.plot(data[-1,:])
#plt.plot(pred)
plt.plot(pred_0)
plt.show()



