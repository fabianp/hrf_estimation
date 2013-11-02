import numpy as np
from scipy import sparse, linalg, optimize
from scipy.sparse import linalg as splinalg
from joblib import Parallel, delayed

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

# .. some auxiliary functions ..
# .. used in optimization ..
def obj(X_, Y_, a, b, u0, size_u, size_v):
    uv0 = khatri_rao(b, a)
    cost = .5 * linalg.norm(Y_ - X_.matvec(uv0), 'fro') ** 2
    return cost

def f(w, X_, Y_, n_task, u0, size_u, size_v):
    W = w.reshape((-1, 1), order='F')
    u, v = W[:size_u], W[size_u:size_u + size_v]
    return obj(X_, Y_, u, v, u0, size_u, size_v)

def fprime(w, X_, Y_, n_task, u0, size_u, size_v):
    n_task = 1
    W = w.reshape((-1, 1), order='F')
    u, v = W[:size_u], W[size_u:size_u + size_v]
    tmp = Y_ - matmat2(X_, u, v, 1)
    grad = np.empty((size_u + size_v, 1))  # TODO: do outside
    grad[:size_u] = rmatmat1(X_, v, tmp, 1)
    grad[size_u:size_u + size_v] = rmatmat2(X_, u, tmp, 1)
    return - grad.reshape((-1,), order='F')


def hess(w, s, X_, Y_, n_task, u0, size_u, size_v):
    # TODO: regularization
    s = s.reshape((-1, 1))
    X_ = splinalg.aslinearoperator(X_)
    size_v = X_.shape[1] / size_u
    W = w.reshape((-1, 1), order='F')
    XY = X_.rmatvec(Y_)  # TODO: move out
    u, v, c = W[:size_u], W[size_u:size_u + size_v], W[size_u + size_v:]
    s1, s2, s3 = s[:size_u], s[size_u:size_u + size_v], s[size_u + size_v:]
    W2 = X_.rmatvec(matmat2(X_, u, v, 1))
    W2 = W2.reshape((-1, s2.shape[0]), order='F')
    XY = XY.reshape((-1, s2.shape[0]), order='F')

    n_task = 1
    A_tmp = matmat2(X_, s1, v, n_task)
    As1 = rmatmat1(X_, v, A_tmp, n_task)
    tmp = matmat2(X_, u, s2, n_task)
    Ds2 = rmatmat2(X_, u, tmp, n_task)

    tmp = matmat2(X_, s1, v, n_task).T

    tmp = matmat2(X_, u, s2, n_task)
    Bs2 = rmatmat1(X_, v, tmp, n_task) + W2.dot(s2) - XY.dot(s2)

    tmp = matmat2(X_, s1, v, n_task)
    Bts1 = rmatmat2(X_, u, tmp, n_task) + W2.T.dot(s1) - XY.T.dot(s1)


    tmp = matmat2(X_, u, s2, n_task)

    line0 = As1 + Bs2
    line1 = Bts1 + Ds2

    out = np.concatenate((line0, line1, line2)).ravel()

    return out


def _rank_one_inner_loop(X, y_i, callback, i, maxiter, method,
                         n_task, rtol, size_u, size_v, u0, verbose, w0):
    X = splinalg.aslinearoperator(X)
    # tmp fix, create zero operator
    w0_i = w0[:, i].ravel('F')
    u0_i = u0[:, i].reshape((-1, 1))
    args = (X, y_i, 1, u0_i, size_u, size_v)
    options = {'maxiter': maxiter, 'xtol': rtol,
               'verbose': verbose}
    if int(verbose) > 1:
        options['disp'] = 5
    out = optimize.minimize(
        f, w0_i, jac=fprime, args=args, hessp=hess,
        method=method, options=options,
        callback=callback)
    if verbose:
        print('Finished problem %s out of %s' % (i + 1, n_task))
        if hasattr(out, 'nit'):
            print('Number of iterations: %s' % out.nit)
        if hasattr(out, 'fun'):
            print('Loss function: %s' % out.fun)
    out = out.x
    W = out.reshape((-1, y_i.shape[1]), order='F')
    ui = W[:size_u].ravel()
    norm_ui = linalg.norm(ui)
    Ui = ui / norm_ui
    Vi = W[size_u:size_u + size_v].ravel() * norm_ui
    Ci = W[size_u + size_v:].ravel()
    return Ui, Vi, Ci


def rank_one(X, Y, size_u, u0=None, v0=None,
             rtol=1e-6, verbose=False, maxiter=1000, callback=None,
             method='TNC', n_jobs=1):
    """
     multi-target rank one model

         ||y - X vec(u v.T) - Z w||^2

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

     rtol : float
         Relative tolerance

     maxiter : int
         maximum number of iterations

     verbose : {0, 1, 2}
         Different levels of verbosity

     Returns
     -------
     U : array, shape (size_u, k)
     V : array, shape (p / size_u, k)
     W : coefficients associated to the drift vectors
     """
    alpha = 0.
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
    w0 = np.zeros((size_u + size_v, n_task))
    w0[:size_u] = u0
    w0[size_u:size_u + size_v] = v0

    U = np.zeros((size_u, n_task))
    V = np.zeros((size_v, n_task))
    C = np.zeros((1, n_task))

    out = Parallel(n_jobs=n_jobs)(
        delayed(_rank_one_inner_loop)(
            X, Y[:, i][:, None], callback, i, maxiter,
            method, n_task, rtol, size_u, size_v, u0, verbose, w0)
        for i in range(Y.shape[1]))

    for i in range(Y.shape[1]):
        tmp = out[i]
        U[:, i] = tmp[0]
        V[:, i] = tmp[1]

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
    u, v, w = rank_one(X.A, y, size_u, verbose=True, rtol=1e-10)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u[:, :1], v[:, :1].T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()
