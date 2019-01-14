import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def rounderino(v,roundSpecs):
    limits = -1*np.flip(np.linspace(roundSpecs[0],roundSpecs[1],2**roundSpecs[2],endpoint=False))
    a = np.convolve(limits,np.array([0.5,0.5]))[1:-1]
    temp = np.searchsorted(a,v,side ='left')
    t = []
    for i in range(temp.size):
        if temp[i] == limits.size:
            t.append(limits[temp[i-1]])
        else:
            t.append(limits[temp[i]])
    return np.asarray(t)
roundSpecs = [-1.5,1.5,8]

def calcZeta(X,X_hat):

    X_hat = X_hat[:,:-1]


    mean_hat = np.mean(X_hat,0)
    mean_ori = np.mean(X,0)

    sigma_hat = np.std(X_hat,0)

    cov = []

    for i in range(X.shape[1]):
        cov.append(1/(X.shape[0]-1)*np.sum((X[:,i]-mean_ori[i])*(X_hat[:,i]-mean_hat[i])))

    cov = np.array(cov)
    zeta = cov/sigma_hat**2
    return zeta

zetas = []
mean_hats = []
mean_oris = []
sigma_hats = []
sigma_oris = []

for file1,file2 in zip(glob.glob('data/X*')[::2],glob.glob('data/X*')[1::2]):
    X = np.load(file2)
    X_hat = np.load(file1)
    zeta = calcZeta(X,X_hat)
    print(file1)
    #np.save('zetas/zeta'+file1[6:8]+'.npy',zeta)

    mean_hats.append(np.mean(X_hat,0))
    mean_oris.append(np.mean(X,0))
    sigma_hats.append(np.std(X_hat,0))
    sigma_oris.append(np.std(X,0))
    zetas.append(zeta)


fig = plt.figure(1)

plt.plot(zetas[0],label = '$\\beta$ = 5')
plt.plot(zetas[1],label = '$\\beta$ = 6')
plt.plot(zetas[2],label = '$\\beta$ = 7')
plt.plot(zetas[3],label = '$\\beta$ = 8')
plt.plot(zetas[-1],label = '$\\beta$ = 16')
plt.xlabel('Iteration (k)')
plt.legend()
plt.ylabel('Estimated correlation gain ($\\zeta^k$)')
plt.grid(1,alpha = 0.5)
plt.savefig('samplecorr.eps')


fig = plt.figure(2) # Means and variances?
i = 0

plt.plot(mean_oris[i], label= 'E[$X^k_i$]')
plt.plot(mean_hats[i][1:],linestyle = ':', label= 'E[$\\hat{X}_i^k$]')

plt.plot(sigma_oris[i], label= 'Var($X^k_i$)')
plt.plot(sigma_hats[i][1:],linestyle = ':', label= 'Var($\\hat{X}_i^k$)')

plt.xlabel('Iteration (k)')
plt.legend()
plt.grid(alpha = .5)
plt.savefig('sample_mean_var.eps')
plt.show()

#plt.plot(calcZeta(X,X_hat))
#plt.plot(X[0,:])
#plt.show()
exit()

'''
data = np.load('Xdata3.npy')
r_data = np.apply_along_axis(rounderino,0,data,roundSpecs)
#print(r_data)
m = np.mean(data,0)
sigma = np.std(data,0)

r_m = np.mean(r_data,0)
r_sigma = np.std(r_data,0)

#sample covanrance
cov = []
r_cov = []
#print(data.shape[0]-1)
for i in range(data.shape[1]-1):
    cov.append(1/(data.shape[0]-1)*np.sum((data[:,i]-m[i])*(data[:,i+1]-m[i+1])))
    r_cov.append(1/(data.shape[0]-1)*np.sum((r_data[:,i]-r_m[i])*(data[:,i+1]-m[i+1])))
    #print(np.sum((data[:,i]-m[i])*(data[:,i+1]-m[i+1])))

#print(((data[:,i]-m[i])*(data[:,i+1]-m[i+1])).shape)
r_cc = r_cov/(r_sigma[:-1]**2)
cc = cov/(sigma[:-1]**2)

pred = m[1:] + r_cc*(r_data[0,:-1]-r_m[:-1])
pred = np.concatenate(([0],pred))

pred2 = np.concatenate(([0],r_data[0,:-1]))

er = data[0,:] - pred
er2 = data[0,:] - pred2

ersum = np.zeros_like(er)
ersum2 = np.zeros_like(er)
ersum3 = np.zeros_like(er)
ersum4 = np.zeros_like(er)

for i in r_data:
    pred = cc*i[:-1]
    pred = np.concatenate(([0],pred))
    #print(ersum.shape,i.shape,pred.shape)
    ersum = ersum + np.absolute(i - pred)

    pred2 = np.concatenate(([0],i[:-1]))
    ersum2 = ersum2 + np.absolute(i - pred2)

    pred3 = np.concatenate(([0],1.02*i[:-1]))
    ersum3 = ersum3 + np.absolute(i - pred3)
    
    pred4 = np.concatenate(([0],.98*i[:-1]))
    ersum4 = ersum4 + np.absolute(i - pred4)

#print(ersum)

ersum = ersum / data.shape[0]
ersum2 = ersum2 / data.shape[0]
ersum3 = ersum3 / data.shape[0]
ersum4 = ersum4 / data.shape[0]


#print(pred)
plt.plot(cov)
plt.plot(sigma**2)
plt.plot(cc)

fig, ax = plt.subplots()
#plt.plot(data[0,:])
plt.title('Et plot for guder!')
ax.plot(ersum2,label = '$\\alpha^k = 1$')
ax.plot(ersum3, label = '$\\alpha^k = 1.02$')
ax.plot(ersum4,label = '$\\alpha^k = 0.98$')
ax.plot(ersum, label = '$\\alpha^k = \\zeta^k$')
plt.xlabel('Iteration [k]')
plt.ylabel('E[$X^{k+1}-\\alpha^kX^k$]')
plt.legend()
plt.grid(1,alpha = 0.5)
axins = zoomed_inset_axes(ax, 5, loc=9) # zoom-factor: 2.5, location: upper-left
axins.plot(ersum2,label = '$\\alpha^k = 1$')
axins.plot(ersum3, label = '$\\alpha^k = 1.02$')
axins.plot(ersum4,label = '$\\alpha^k = 0.98$')
axins.plot(ersum, label = '$\\alpha^k = \\rho^k$')
plt.grid(1,alpha = 0.5)
x1, x2, y1, y2 = 4, 8, 0.135, 0.155 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
#plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

#plt.savefig('correr.eps')

fig.patch.set_facecolor((207/255,228/255,242/255))

plt.savefig('correr_poster.eps',facecolor=fig.get_facecolor())

f3 = plt.figure(3)
plt.plot(m)
plt.xlabel('Iteration [k]')
plt.ylabel('E[$X_i^k$]')
plt.grid(1,alpha = 0.5)

#plt.savefig('samplemean.eps')

f3 = plt.figure(4)
plt.plot(sigma**2, label = 'Var($X_i^k$)')
plt.plot(cc,label = '$\\zeta^k$')
plt.xlabel('Iteration [k]')
plt.legend()
plt.grid(1,alpha = 0.5)

#plt.savefig('samplevar.eps')
#np.save('alphax.npy',cc)

plt.show()
print(data)
'''