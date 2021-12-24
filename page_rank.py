import numpy as np
import numpy.linalg as la
from readonly.PageRankFunctions import *
np.set_printoptions(suppress=True)

# the PageRank for an arbitrarily sized internet.
# The functions inputs are the linkMatrix, and d the damping parameter
# Power-Iteration
def pageRank(linkMatrix, d) :
    n = linkMatrix.shape[0]
    M = d * linkMatrix + (1-d)/n * np.ones([n, n])
    r = 100*np.ones(n)/n #initial guess
    new_r = M@r
    i =0
    while la.norm(r - new_r) > 0.01 :
        r = new_r
        new_r = M @ r
        i+=1
    print(str(i) + " iterations to convergence.")
    return r
