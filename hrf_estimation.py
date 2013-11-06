import numpy as np
from scipy import sparse, linalg, optimize
from scipy.sparse import linalg as splinalg
from joblib import Parallel, delayed, cpu_count

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
def obj(X_, Y_, u, v, size_u, size_v, alpha, u0):
    uv0 = khatri_rao(v, u)
    cost = .5 * linalg.norm(Y_ - X_.matvec(uv0), 'fro') ** 2
    if alpha != 0.:
        cost += 0.5 * (linalg.norm(u - u0) ** 2)
    return cost

def f(w, X_, Y_, n_task, size_u, size_v, alpha, u0):
    W = w.reshape((-1, 1), order='F')
    u, v = W[:size_u], W[size_u:size_u + size_v]
    return obj(X_, Y_, u, v, size_u, size_v, alpha, u0)

def fprime(w, X_, Y_, n_task, size_u, size_v, alpha, u0):
    n_task = 1
    W = w.reshape((-1, 1), order='F')
    u, v = W[:size_u], W[size_u:size_u + size_v]
    tmp = Y_ - matmat2(X_, u, v, 1)
    grad = np.empty((size_u + size_v, 1))  # TODO: do outside
    grad[:size_u] = rmatmat1(X_, v, tmp, 1) + alpha * (u - u0)
    grad[size_u:size_u + size_v] = rmatmat2(X_, u, tmp, 1)
    return - grad.reshape((-1,), order='F')


def hess(w, s, X_, Y_, n_task, size_u, size_v, alpha, u0):
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


def _rank_one_inner_loop(X, y_i, callback, maxiter, method,
                         n_task, rtol, size_u, size_v, verbose, w0, alpha, u0):
    X = splinalg.aslinearoperator(X)
    n_task = y_i.shape[1]
    w0 = np.random.randn((size_u + size_v))
    U = []
    V = []
    for i in range(n_task):
        # import ipdb; ipdb.set_trace()
        args = (X, y_i[:, i, np.newaxis], 1, size_u, size_v, alpha, u0)
        options = {'maxiter': maxiter, 'verbose': verbose}
        if int(verbose) > 1:
            options['disp'] = 5
        kwargs = {}
        if method == 'Newton-CG':
            kwargs['hessp'] = hess
        out = optimize.minimize(
            f, w0, jac=fprime, args=args,
            method=method, options=options,
            callback=callback, tol=rtol, **kwargs)
        assert out.success
        if verbose:
            print('Finished problem %s out of %s' % (i + 1, n_task))
            if hasattr(out, 'nit'):
                print('Number of iterations: %s' % out.nit)
            if hasattr(out, 'fun'):
                print('Loss function: %s' % out.fun)
        w0[:] = out.x.copy() # use as warm 
        ui = out.x[:size_u].ravel()
        vi = out.x[size_u:size_u + size_v].ravel()
        norm = linalg.norm(ui)
        U.append(ui / norm)
        V.append(vi * norm)
        w0[:size_u] /= norm
        w0[size_u:] *= norm
    return U, V 


def rank_one(X, Y, size_u, can_hrf, alpha=0., u0=None,
             rtol=1e-6, verbose=False, maxiter=1000, callback=None,
             method='L-BFGS-B', n_jobs=1):
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

    method: {'L-BFGS-B', 'TNC', 'CG'}
        Different solvers. All should yield the same result but their efficiency
        might vary.

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

    u0 = np.ones((size_u, 1))
    if u0.ndim == 1:
        u0 = u0[:, None]

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

    if n_jobs == -1:
        n_jobs = cpu_count()
    Y_split = np.array_split(Y, n_jobs, axis=1)

    out = Parallel(n_jobs=n_jobs)(
        delayed(_rank_one_inner_loop)(
            X, y_i, callback, maxiter,
            method, n_task, rtol, size_u, size_v, verbose, w0, alpha, u0)
        for y_i in Y_split)

    counter = 0
    for tmp in out:
        u, v = tmp
        for i in range(len(u)):
            U[:, counter] = u[i]
            V[:, counter] = v[i]
            counter += 1

    # normalize
    sign = np.sign(U.T.dot(can_hrf))
    norm = sign * np.sqrt((U * U).sum(0)) # norming twice ?
    return U / norm, V * norm



if __name__ == '__main__':
    size_u, size_v = 9, 48
    X = sparse.csr_matrix(np.random.randn(100, size_u * size_v))
    Z = np.random.randn(1000, 20)
    u_true, v_true = np.random.rand(size_u, 2), 1 + .1 * np.random.randn(size_v, 2)
    B = np.dot(u_true, v_true.T)
    y = X.dot(B.ravel('F')) + .1 * np.random.randn(X.shape[0])
    #y = np.array([i * y for i in range(1, 3)]).T
    u, v = rank_one(X.A, y, size_u, np.random.randn(size_u),
                    verbose=True, rtol=1e-10)

    import pylab as plt
    plt.matshow(B)
    plt.title('Groud truth')
    plt.colorbar()
    plt.matshow(np.dot(u[:, :1], v[:, :1].T))
    plt.title('Estimated')
    plt.colorbar()
    plt.show()
