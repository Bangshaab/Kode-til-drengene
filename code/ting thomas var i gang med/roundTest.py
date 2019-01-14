import numpy as np

lowerBound = -1
upperBound = 1
nBits = 7


def rounderino(lower,upper,nBits,v):
    limits = np.linspace(lowerBound,upperBound,2**nBits)
    a = np.convolve(limits,np.array([0.5,0.5]))[1:-1]
    temp = np.searchsorted(a,v,side ='left')
    t = []
    print(temp)
    
    for i in range(temp.size):
        if temp[i] == limits.size:
            t.append(limits[temp[i-1]])
        else:
            t.append(limits[temp[i]])

    return np.asarray(t)




x = np.array([-2,-0.01,0,0.5,1.1])

print('Before sort: ',x)

x = rounderino(-1,1,nBits,x)
print('After sort: ',x)
