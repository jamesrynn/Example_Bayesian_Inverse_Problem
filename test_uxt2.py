import numpy as np
import util


x = np.array([0.2, 0.4, 0.6, 0.8])

t = np.array([0.3, 0.6, 0.890])

lam = np.array([1,2,3,4,5])


#U1 = np.zeros((4,3,5))
#for i,j in enumerate(lam):
#	U1[i,:,:] = util.uxt(x,t,j)


U2 = util.uxt2(x,t,lam)

#U1==U2

#print(U1.shape)
print(U2.shape)

#print(U1)
print(U2)


