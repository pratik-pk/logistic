import numpy as np
import matplotlib.pyplot as plt
N=100
D=2
X=np.random.randn(N,D)
X[:50,:]=X[:50,:]-2*np.ones((50,D))
X[50:,:]=X[50:,:]+2*np.ones((50,D))
T=np.array([0]*50+[1]*50)
ones=np.array([[1]*N]).T
Xb=np.concatenate((ones,X),axis=1)
def sigmoid(z):
    return 1/(1+np.exp(-z))
w=np.array([0,4,4])
plt.scatter(X[:,0],X[:,1], c=T,s=100,alpha=0.5)
X_axis=np.linspace(-6,-6,100)
Y_axis=-X_axis
plt.plot(X_axis,Y_axis)
plt.show()
