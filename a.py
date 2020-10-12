import numpy as np

x = np.random.rand(2)
w = np.random.rand(2,3)
b = np.random.rand(3)

x = np.array([3,5])
print(w)
print(b)
y = np.dot(x,w)+b
print(y)