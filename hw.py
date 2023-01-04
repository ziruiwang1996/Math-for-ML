import numpy as np
import numpy.linalg
from numpy.linalg import inv
import math
a = np.array([[3/2, -1],
             [-1/2, 1/2]])
c = np.array([[-1-math.sqrt(3), -1+math.sqrt(3)],
             [1, 1]])
d = a@a
#print(d)

M = np.array([[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0]])
d = 0.2
vals, vecs = np.linalg.eig(M)
print(vals)
print(vecs)
