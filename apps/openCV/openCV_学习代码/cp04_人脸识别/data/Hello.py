import numpy as np

a = np.array([1,2,3])

print(np.repeat(a,3))

b = np.array([[1,2,3],[6,7,8]])
print(b)
print(np.repeat(b,3)) # reshape
c = np.repeat(b,3,axis = 0)
d = np.repeat(c, 3, axis=1)
print(d)