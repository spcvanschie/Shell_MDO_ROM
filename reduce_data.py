import numpy as np
from pymor.basic import *
import matplotlib.pyplot as plt

num_snaps = 6
n_dof = 5904
X = np.empty((n_dof,num_snaps))

#assemble snapshot matrix from 
for i in range(num_snaps):
    filename = '/home/jkrokowski/Documents/Research/Shell_MDO_ROM/Shell_MDO_ROM/snapshots/disp_' + str(i) + '.npy'
    X[:,i] = np.load(filename)

print(X.shape)
# U,S,V = np.linalg.svd(X@X.T)
# print(U.shape)
# print(S.shape)
# print(V.shape)
# print(S.shape)
X_mean = np.empty_like(X)
for i in range(n_dof):
    X_mean[i,:] = X[i,:] - np.mean(X[i,:])

U,S,V = np.linalg.svd(X_mean)
print(S)
fig,ax = plt.subplots(nrows=1,ncols=1)
fig.suptitle('Singular Values of Snapshot matrix')
ax.set_xlabel('Singular Value index')
ax.set_ylabel('Singular Value (of mean subtracted snapshot matrix)')
ax.semilogy(S)
plt.show()

X_va = NumpyVectorSpace(n_dof).from_numpy(X.T)

pod_basis,pod_singular_values = pod(X_va)
print(pod_singular_values)
print(pod_basis)
