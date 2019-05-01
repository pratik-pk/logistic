import numpy as np
N=100
D=2
X=np.random.randn(N,D)
#center the first 50 points at (-2,-2)
X[:50,:]=X[:50,:]-2*np.ones((50,D))
#center the last %0 points at (2,2)
X[50:,:]=X[50:,:]+2*np.ones((50,D))
#labels:first 50 are 0, last 50 are 1

T=np.array([0]*50+[1]*50)
ones=np.array([[1]*N]).T
Xb=np.concatenate((ones,X),axis=1)
w=np.random.randn(D+1)
z=Xb.dot(w)
def sigmoid(z):
    return 1/(1+np.exp(-z))
y=sigmoid(z)

def cross_entropy(T,y):
    E=0
    for i in range(N):
        if T[i]==1:
            E-=np.log(y[i])
        else:
            E-=np.log(1-y[i])
    return E
print(cross_entropy(T,y))
w=np.array([0,4,4])
z=Xb.dot(w)
y=sigmoid(z)
print(f'second val : {cross_entropy(T,y)}')
