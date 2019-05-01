import numpy as np
N=100
D=2
X=np.random.randn(N,D)
X[:50,:]=X[:50,:]-2*np.ones((50,D))
X[50:,:]=X[50:,:]+2*np.ones((50,D))
T=np.array([0]*50+[1]*50)
ones=np.array([[1]*N]).T
Xb=np.concatenate((ones,X),axis=1)
w=np.random.randn(D+1)
Z=Xb.dot(w)
def sigmoid(z):
    return 1/(1+np.exp(-z))
y=sigmoid(Z)

def cross_entropy(T,y):
    E=0
    for i in range(N):
        if(T[i]==1):
            E-=np.log(y[i])
        else:
            E-=np.log(1-y[i])
    return E
print(cross_entropy(T,y))

learning_rate=0.01
for i in range(100):
    if i%10==0:
        print(cross_entropy(T,y))
    w+=learning_rate*np.dot((T-y).T,Xb)
    y=sigmoid(Xb.dot(w))
print(f'final {w}')

