import numpy as np
from scipy.sparse import linalg as splinalg
from scipy import sparse, linalg, optimize
import rank_one_

def obo_analysis_matrix(obo_matrices, l2_penalty=0.0):

    fir_length = obo_matrices.shape[2] / 2

    output_evt = np.zeros([len(obo_matrices), fir_length,
                       obo_matrices.shape[1]])
    output_rest = output_evt.copy()

    for obo_mat, out_evt, out_rest in zip(
            obo_matrices, output_evt, output_rest):
        gram = obo_mat.T.dot(obo_mat)

        tmp = np.linalg.pinv(gram +
                l2_penalty * np.eye(len(gram))).dot(obo_mat.T)

        out_evt[:] = tmp[:fir_length]
        out_rest[:] = tmp[fir_length:]

    output = (output_evt.reshape(len(obo_matrices) * fir_length,
                          obo_matrices.shape[1]),
              output_rest.reshape(len(obo_matrices) * fir_length,
                          obo_matrices.shape[1]))
    return output


# @profile
def rank_one(X_obo_fir, Y, u_0=None, XTY=None, X_gram=None,
             max_iter=3, rtol=1e-6):
    """
    Will perform alternate minimization on activation map and hrf for
    a group of Y.shape[1] voxels using the one by one scheme. Can learn
    one global hrf or one for event and one for non-event.

    Will use the same HRF for all given voxels.
    """

    fir_length = X_obo_fir.shape[2] / 2
    if u_0 is None:
        u_0 = np.ones([1, fir_length])
    else:
        u_0 = np.atleast_2d(u_0)
        assert u_0.shape[1] == fir_length

    num_hrfs = len(u_0)
    if num_hrfs not in [1, 2]:
        raise Exception("Can only estimate one or two hrfs")

    u = u_0.copy()
    counter = 0
    improvement = np.inf

    # precompute stuff
    if (X_gram is None) or (XTY is None):
        X_gram, XTY = _pre_calc_X_gram_and_XTY(X_obo_fir, Y)

    while (counter < max_iter) and (improvement > rtol):
        counter += 1

        # estimate activity
        X_obo = _apply_hrf(X_obo_fir, u)

        X_obo_pinv, X_obo_rest_pinv = obo_analysis_matrix(X_obo)

        betas, betas_rest = [m.dot(Y) for m in [X_obo_pinv, X_obo_rest_pinv]]


        all_betas = np.array([betas, betas_rest])
        # estimate hrf
        # u = _estimate_hrf(X_obo_fir, betas, betas_rest, Y,
        #                   same_hrf=num_hrfs == 1)

        u = _estimate_hrf(all_betas, X_gram, XTY, num_hrfs == 1)


    return u, betas, betas_rest






def _apply_hrf(X, hrf):
    """Applies given hrf to given FIR design"""
    if len(hrf) == 1:
        hrf = np.concatenate([hrf] * 2)

    fir_length = hrf.shape[1]

    X = X.reshape([X.shape[0], X.shape[1], 2, fir_length])

    return np.einsum('ijkl, kl -> ijk', X, hrf)


def _pre_calc_X_gram_and_XTY(X, Y):

    # n_voxels = Y.shape[1]
    # fir_length = X.shape[-1] / 2

    X_gram = np.einsum('ijk, ijl -> ikl', X, X)

    XTY = np.einsum('ijk, jl -> ikl', X, Y)

    return X_gram, XTY


def _estimate_hrf(all_betas, X_gram, XTY, same_hrf=True):

    beta_gram = np.einsum('ijk, ljk -> ilj', all_betas, all_betas)

    fir_length = XTY.shape[1] / 2

    X_gram = X_gram.reshape(X_gram.shape[0], 2, fir_length, 2, fir_length)

    beta_X_gram = np.einsum('ijk, kiljm -> iljm', beta_gram, X_gram)

    XTY = XTY.reshape(XTY.shape[0], 2, fir_length, XTY.shape[-1])

    beta_XTY = np.einsum('ijk, jilk -> il', all_betas, XTY)

    if same_hrf:
        beta_XTY = beta_XTY.sum(0)
        beta_X_gram = beta_X_gram.sum(2).sum(0)

        return np.linalg.pinv(beta_X_gram).dot(beta_XTY).reshape(1, -1)
    else:
        beta_X_gram = beta_X_gram.reshape(2 * fir_length, 2 * fir_length)
        beta_XTY = beta_XTY.reshape(2 * fir_length)

        return np.linalg.pinv(beta_X_gram).dot(beta_XTY).reshape(2, fir_length)


def _compute_obo(x0, X_bd, X_all, yi, plot, size_u, size_v, verbose,
                 maxiter, X):
    X_bd_ = splinalg.aslinearoperator(X_bd)
    res_bd = sparse.block_diag([np.ones((yi.shape[0], 1))] * size_v).tocsr()

    def fun_grad(w):
        u_i = w[:size_u]
        v_i = w[size_u:size_u + size_v]
        w_i = w[size_u + size_v:size_u + 2 * size_v]
        Xuv = rank_one_.Xba(X_bd_, u_i[:, None], v_i[:, None], 1)
        Xuv = Xuv.reshape((yi.shape[0], -1), order='F')
        Xuw = rank_one_.Xba(X_bd_, u_i[:, None], w_i[:, None], 1)
        Xuw = Xuw.reshape((yi.shape[0], -1), order='F')
        X_all_u = X_all.dot(u_i)
        X_all_uw = X_all_u[:, None] * w_i
        res = yi[:, None] - Xuv + Xuw - X_all_uw
        res = np.asarray(res)  # matrix type, go wonder
        res_bd.data = res.ravel('F')
        X_res = X_bd_.rmatvec(res_bd)
        X_res = X_res.sum(1).reshape((size_u, size_v), order='F')

        X_all_res = - X_all.T.dot(res)
        grad_u = - X_res.dot(v_i) + X_res.dot(w_i) + X_all_res.dot(w_i)
        grad_v = - X_res.T.dot(u_i)
        grad_w = grad_v + X_all_res.T.dot(u_i)
        grad = np.concatenate((grad_u, grad_v, grad_w), axis=1)
        return 0.5 * (res * res).sum(), np.asarray(grad).ravel()


    out = optimize.fmin_tnc(fun_grad, x0, disp=5, maxfun=maxiter, messages=verbose)
    u = out[0][:size_u]
    u = u * np.sign(u[5])
    u /= linalg.norm(u)
    if plot:
        import pylab as pl
        pl.plot(u)
        pl.draw()
    v = out[0][size_u:size_u + size_v]
    return u, v

def rank_one_obo(X, Y, size_u, u0=None, rtol=1e-3,
                 maxiter=300, verbose=False,
                 callback=None, v0=None, plot=False, n_jobs=1):
    """
    multi-target rank one model

        ||y - X vec(u v.T)||_2 ^2

    TODO: prior_u

    Parameters
    ----------
    X : array-like, shape (n, p)
    Y_train : array-like, shape (n, k)
    size_u : integer
        Must be divisor of p
    u0 : array
        Initial value for u
    v0 : array
        Initial value for v
    rtol : float
    maxiter : int
        maximum number of iterations
    verbose : bool
        If True, prints the value of the objective
        function at each iteration

    Returns
    -------
    U : array, shape (size_u, k)
    V : array, shape (p / size_u, k)
    W : XXX

    Reference
    ---------
    """


    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]
    size_v = X.shape[1] / size_u

    # .. check dimensions in input ..
    if X.shape[0] != Y.shape[0]:
        raise ValueError('Wrong shape for X, y')

    if plot:
        import pylab as pl

    if u0 is None:
        u0 = np.random.randn(size_u, 1)
    if u0.ndim == 1 or u0.shape[1] == 1:
        u = np.empty((u0.size, n_task))
        u[:, :] = u0
        u0 = u

    if v0 is None:
        v0 = np.random.randn(size_v, n_task)

    w0 = np.random.randn(size_v, n_task)
    if plot:
        fig = pl.figure()
        pl.show()

    II = sparse.hstack([sparse.eye(size_u)] * size_v)
    X_all = X.dot(II.T)
    X_blockdiag = []
    for i in range(size_v):
        X_tmp = X[:, size_u * i:size_u * (i + 1)]
        X_blockdiag.append(X_tmp)
    X_bd = sparse.block_diag(X_blockdiag).tocsr()
    x0 = np.concatenate((u0, v0, w0))

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=n_jobs)(
        delayed(_compute_obo)(
            x0[:, k], X_bd, X_all, Y[:, k],
            plot, size_u, size_v, verbose, maxiter, X) for k in range(n_task))
    U = np.array([out[i][0] for i in range(n_task)]).T
    V = np.array([out[i][1] for i in range(n_task)]).T
    return U, V