"""
Canonical Polyadic Decomposition (aka PARAFAC/CANDECMP/CP) tensor decomposition
"""

import numpy as np
from scipy import linalg

def khatri_rao(A, B):
    """
    Compute the Khatri-rao product, where the partition is taken to be
    the vectors along axis one.

    This is a helper function for rank_one

    Parameters
    ----------
    A : array, shape (n, p)
    B : array, shape (m, p)
    AB : array, shape (nm, p), optimal
        if given, result will be stored here

    Returns
    -------
    a*b : array, shape (nm, p)
    """
    num_targets = A.shape[1]
    assert B.shape[1] == num_targets
    return (A.T[:, :, np.newaxis] * B.T[:, np.newaxis, :]
    ).reshape(num_targets, len(B) * len(A)).T


def cpd_als(X, trank, orthogonality=[False, False, False], maxiter=500,
                tol=1e-3,
                verbose=0):
    """
    PARAFAC with orthogonality constraints in the first mode

    Parameters
    ----------
    X : array-like, shape (n, m, p)
    trank: tensor rank

    Returns
    -------
    """
    # The different views we will need
    X = np.asarray(X)
    assert len(orthogonality) == 3
    assert len(X.shape) == 3
    n, m, k = X.shape
    X1 = X.reshape((n, m * k), order='F')
    X2 = np.rollaxis(X, 1).reshape((m, k * n), order='F')
    X3 = np.rollaxis(X, 2).reshape((k, n * m), order='F')

    A0 = np.random.randn(n, trank)
    B0 = np.random.randn(m, trank)
    C0 = np.random.randn(k, trank)
    en = linalg.norm(X1 - A0.dot(khatri_rao(C0, B0).T), 'fro')
    norm_X = linalg.norm(X1, 'fro')
    energy = [en]
    for i in range(maxiter):
        M = khatri_rao(C0, B0).T.dot(X1.T)
        if orthogonality[0]:
            U0, s, V0t = linalg.svd(M)
            A0 = (U0.dot(np.eye(trank, V0t.shape[0])).dot(V0t)).T
        else:
            A0 = linalg.lstsq(khatri_rao(C0, B0), X1.T)[0].T
        if orthogonality[1]:
            raise NotImplementedError
        tmp = khatri_rao(C0, A0)
        B0 = linalg.lstsq(tmp, X2.T)[0].T
        if orthogonality[2]:
            raise NotImplementedError
        tmp = khatri_rao(B0, A0)
        C0 = linalg.lstsq(tmp, X3.T)[0].T
        en = linalg.norm(X1 - A0.dot(khatri_rao(C0, B0).T), 'fro')
        energy.append(en)
        if np.abs(energy[-2] - energy[-1]) < tol:
            break
        if (verbose > 0): # and (i % 100 == 0):
            explained_variance = (1 - ((en / norm_X) ** 2))
            print('.. CPD explained variance: %.04f, iter %s ..' % (explained_variance, i))
        elif (verbose > 1) and (i % 10 == 0):
            explained_variance = (1 - ((en / norm_X) ** 2))
            print('.. CPD explained variance: %.04f, iter %s' % (explained_variance, i))
    if verbose > 1:
        explained_variance = (1 - ((en / norm_X) ** 2))
        print('.. final explained variance: %.04f, iter %s ..' % (explained_variance, i))
    return A0, B0, C0, np.array(energy)

if __name__ == '__main__':
    n, m, p = 10, 2, 6
    X = np.random.randn(n, m * p)
    X = X.reshape((n, m, p))
    A, B, C, energy = cpd_als(X, m * p)

    import pylab as pl
    pl.plot(np.arange(energy.size), energy - np.min(energy), lw=4)
    pl.yscale('log')
    pl.show()
