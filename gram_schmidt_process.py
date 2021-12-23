import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14

def gsBasis4(A) :
    B = np.array(A, dtype=np.float_)
    # normalise the zeroth column
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    # subtract any overlap with  new zeroth vector.
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    # If there's anything left after that subtraction, then B[:, 1] is linearly independant of B[:, 0]
    # If this is the case, we can normalise it. Otherwise we'll set that vector to zero.
    if la.norm(B[:, 1]) > verySmallNumber :
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else :
        B[:, 1] = np.zeros_like(B[:, 1])
    # column 2.
    B[:,2] = B[:,2] - B[:,2]@B[:,0]*B[:,0]
    B[:,2] = B[:,2] - B[:,2]@B[:,1]*B[:,1]
    if la.norm(B[:, 2]) > verySmallNumber :
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else:
        B[:, 2] = np.zeros_like(B[:, 2])
    # column 3
    B[:,3] = B[:,3] - B[:,3]@B[:,0]*B[:,0]
    B[:,3] = B[:,3] - B[:,3]@B[:,1]*B[:,1]
    B[:,3] = B[:,3] - B[:,3]@B[:,2]*B[:,2]
    if la.norm(B[:, 3]) > verySmallNumber :
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else:
        B[:, 3] = np.zeros_like(B[:, 3])

    return B

# generalise the procedure.
def gsBasis(A) :
    B = np.array(A, dtype=np.float_)
    for i in range(B.shape[1]) :
        for j in range(i) :
            B[:, i] = B[:, i] - B[:, i]@B[:, j]*B[:, j]
        if la.norm(B[:, i])> verySmallNumber:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])
    return B

# This function uses the Gram-schmidt process to calculate the dimension
# spanned by a list of vectors.
# Since each vector is normalised to one, or is zero,
# the sum of all the norms will be the dimension.
def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))
