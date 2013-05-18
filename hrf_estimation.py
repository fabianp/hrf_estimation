import numpy as np
from scipy import sparse, linalg, optimize
from scipy.sparse import linalg as splinalg

__version__ = '0.3'

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

def matmat2(X, a, b, n_task):
    """
    X (b * a)
    """
    uv0 = khatri_rao(b, a)
    return X.matvec(uv0)


def rmatmat1(X, a, b, n_task):
    """
    (I kron a^T) X^T b
    """
    b1 = X.rmatvec(b[:X.shape[0]]).T
    B = b1.reshape((n_task, -1, a.shape[0]), order='F')
    res = np.einsum("ijk, ik -> ij", B, a.T).T
    return res


def rmatmat2(X, a, b, n_task):
    """
    (a^T kron I) X^T b
    """
    b1 = X.rmatvec(b).T
    B = b1.reshape((n_task, -1, a.shape[0]), order='C')
    tmp = np.einsum("ijk, ik -> ij", B, a.T).T
    return tmp

def rank_one(X, Y, alpha, size_u, u0=None, v0=None, Z=None, rtol=1e-6, verbose=False, maxiter=1000):
    """
    multi-target rank one model

        ||y - X vec(u v.T) - Z w||^2 + alpha * ||u - u_0||^2

    Parameters
    ----------
    X : array-like, sparse matrix or LinearOperator, shape (n, p)
        The design matrix

    Y : array-lime, shape (n, k)
        Time-series vector. Several time-series vectors can be given at once,
        however for large system becomes unstable. We do not recommend
        using more than k > 100.

    size_u : integer
        Must be divisor of p

    u0 : array

    Z : array, sparse matrix or LinearOperator, shape (n, q)
        Represents the drift vectors.

    rtol : float
        Relative tolerance

    maxiter : int
        maximum number of iterations

    verbose : boolean

    Returns
    -------
    U : array, shape (size_u, k)
    V : array, shape (p / size_u, k)
    W : coefficients associated to the drift vectors
    """

    X = splinalg.aslinearoperator(X)
    if Z is None:
        # create identity operator
        Z_ = splinalg.LinearOperator(shape=(X.shape[0], 1),
                                     matvec=lambda x: np.zeros((X.shape[0], x.shape[1])),
                                     rmatvec=lambda x: np.zeros((1, x.shape[1])), dtype=np.float)
    else:
        Z_ = splinalg.aslinearoperator(Z)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    if u0 is None:
        u0 = np.ones((size_u, n_task))
    if u0.size == size_u:
        u0 = u0.reshape((-1, 1))
        u0 = np.repeat(u0, n_task, axis=1)
    if v0 is None:
        v0 = np.ones(X.shape[1] / size_u * n_task)  # np.random.randn(shape_B[1])

    size_v = X.shape[1] / size_u
    #u0 = u0.reshape((-1, n_task))
    v0 = v0.reshape((-1, n_task))
    w0 = np.zeros((size_u + size_v + Z_.shape[1], n_task))
    w0[:size_u] = u0
    w0[size_u:size_u + size_v] = v0
    w0 = w0.reshape((-1,), order='F')

    # .. some auxiliary functions ..
    # .. used in conjugate gradient ..
    def obj(X_, Y_, Z_, a, b, c, u0):
        uv0 = khatri_rao(b, a)
        cost = .5 * linalg.norm(Y_ - X_.matvec(uv0) - Z_.matmat(c), 'fro') ** 2
        reg = alpha * linalg.norm(a - u0, 'fro') ** 2
        return cost + reg

    def f(w, X_, Y_, Z_, n_task, u0):
        W = w.reshape((-1, n_task), order='F')
        u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
        return obj(X_, Y_, Z_, u, v, c, u0)

    def fprime(w, X_, Y_, Z_, n_task, u0):
        W = w.reshape((-1, n_task), order='F')
        u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
        tmp = Y_ - matmat2(X_, u, v, n_task) - Z_.matmat(c)
        grad = np.empty((size_u + size_v + Z_.shape[1], n_task))  # TODO: do outside
        grad[:size_u] = rmatmat1(X, v, tmp, n_task) - alpha * (u - u0)
        grad[size_u:size_u + size_v] = rmatmat2(X, u, tmp, n_task)
        grad[size_u + size_v:] = Z_.rmatvec(tmp)
        return - grad.reshape((-1,), order='F')

    Y_split = [Y] #np.array_split(Y, n_split, axis=1)
    U = np.zeros((size_u, n_task))
    V = np.zeros((size_v, n_task))
    C = np.zeros((Z_.shape[1], n_task))
    counter = 0
    for y_i in Y_split: # TODO; remove
        w0_i = w0.reshape((size_u + size_v + Z_.shape[1], n_task), order='F')[:, counter:(counter + y_i.shape[1])]
        u0_i = u0[:, counter:(counter + y_i.shape[1])]
        out = optimize.fmin_l_bfgs_b(f, w0_i, fprime=fprime, factr=rtol / np.finfo(np.float).eps,
                                         args=(X, y_i, Z_, y_i.shape[1], u0_i), maxfun=maxiter)
        if out[2]['warnflag'] != 0:
            print('Not converged')
        W = out[0].reshape((-1, y_i.shape[1]), order='F')
        U[:, counter:counter + y_i.shape[1]] = W[:size_u]
        V[:, counter:counter + y_i.shape[1]] = W[size_u:size_u + size_v]
        C[:, counter:counter + y_i.shape[1]] = W[size_u + size_v:]
        counter += y_i.shape[1]
        if verbose:
            print('Completed %.01f%%' % ((100. * counter) / Y.shape[1]))

    if Z is None:
        return U, V
    else:
        return U, V, C


if __name__ == '__main__':
    size_u, size_v = 9, 48
    X = sparse.csr_matrix(np.random.randn(100, size_u * size_v))
    Z = np.random.randn(1000, 20)
    u_true, v_true = np.random.rand(size_u, 2), 1 + .1 * np.random.randn(size_v, 2)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .1 * np.random.randn(X.shape[0])
    #y = np.array([i * y for i in range(1, 3)]).T
    u, v, w = rank_one(X.A, y, .1, size_u, Z=np.random.randn(X.shape[0], 3), verbose=True, rtol=1e-10)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u[:, :1], v[:, :1].T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()