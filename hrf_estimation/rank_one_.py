import numpy as np
from scipy import sparse, linalg, optimize
from joblib import Parallel, delayed, cpu_count, Memory
import warnings

# local imports
from . import hrf
from . import utils
from .utils import create_design_matrix


def IaXb(X, a, b):
    """
    (I kron a^T) X^T b
    """
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    b1 = X.T.dot(b[:X.shape[0]]).T
    B = b1.reshape((1, -1, a.shape[0]), order='F')
    res = np.einsum("ijk, ik -> ij", B, a.T).T
    return res


def aIXb(X, a, b):
    """
    (a^T kron I) X^T b
    """
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    b1 = X.T.dot(b).T
    B = b1.reshape((1, -1, a.shape[0]), order='C')
    tmp = np.einsum("ijk, ik -> ij", B, a.T).T
    return tmp


# .. some auxiliary functions ..
# .. used in optimization ..

def f_r1(w, X, Y, size_u, size_v):
    """Objective function for the rank-one FIR model"""
    u, v, bias = w[:size_u], w[size_u:size_u + size_v], w[size_u + size_v:]
    res = Y.ravel() - X.dot(np.outer(u, v).ravel('F')).ravel() - bias
    cost = .5 * linalg.norm(res) ** 2
    cost -= .5 * (linalg.norm(u) ** 2)
    return cost


def fprime(w, X, Y, drifts, size_u, size_v):
    u, v, bias = w[:size_u], w[size_u:size_u + size_v], w[size_u + size_v:]
    res = Y.ravel() - X.dot(np.outer(u, v).ravel('F')).ravel() - drifts.dot(bias)
    grad = np.empty((size_u + size_v + drifts.shape[1]))
    grad[:size_u] = IaXb(X, v, res).ravel() + u
    grad[size_u:size_u + size_v] = aIXb(X, u, res).ravel()
    grad[size_u + size_v:] = drifts.dot(res)
    return - grad

def f_grad(w, X, Y, drifts, size_u, size_v):
    """Returns function AND gradient of the rank-one model"""
    u, v, bias = w[:size_u], w[size_u:size_u + size_v], w[size_u + size_v:]
    assert len(bias) == drifts.shape[1]
    res = Y.ravel() - X.dot(np.outer(u, v).ravel('F')).ravel() - drifts.dot(bias)
    cost = .5 * linalg.norm(res) ** 2
    grad = np.empty((size_u + size_v + drifts.shape[1]))
    grad[:size_u] = IaXb(X, v, res).ravel()
    grad[size_u:size_u + size_v] = aIXb(X, u, res).ravel()
    grad[size_u + size_v:] = drifts.T.dot(res)
    return cost, -grad

def f_grad_betas(w, X, Y, drifts, size_u, size_v):
    """Returns the function and gradient of the rank-one model
    assumes the HRF is fixed and only optimizes for beta
    """
    u, v, bias = w[:size_u], w[size_u:size_u + size_v], w[size_u + size_v:]
    assert len(bias) == drifts.shape[1]
    res = Y.ravel() - X.dot(np.outer(u, v).ravel('F')).ravel() - drifts.dot(bias)
    cost = .5 * linalg.norm(res) ** 2
    cost -= .5 * (linalg.norm(u) ** 2)
    grad = np.empty((size_u + size_v + drifts.shape[1]))
    grad[:size_u] = 0
    grad[size_u:size_u + size_v] = aIXb(X, u, res).ravel()
    grad[size_u + size_v:] = drifts.T.dot(res)
    return cost, -grad


def f_separate(w, X, Y, size_u, size_v, X_all):
    # for the GLM with separate designs
    u, v, z = w[:size_u], w[size_u:size_u + size_v], w[size_u + size_v:]
    norm = 0
    uv = np.outer(u, v)
    uz = np.outer(u, z)
    X_all_uz = X_all.dot(uz)
    tmp = uv - uz
    for j in range(size_v):  # iterate through trials
        Xi = X[:, j*size_u:(j+1)*size_u]
        res_i = Y - Xi.dot(tmp[:, j]) - X_all_uz[:, j]
        assert res_i.ndim == 1
        norm += .5 * linalg.norm(res_i) ** 2
    norm -= .5 * (linalg.norm(u) ** 2)
    return norm


def fprime_separate(w, X, Y, size_u, size_v, X_all):
    u, v, z = w[:size_u], w[size_u:size_u + size_v], w[size_u + size_v:]
    grad = np.zeros((size_u + 2 * size_v))
    uv = np.outer(u, v)
    uz = np.outer(u, z)
    X_all_uz = X_all.dot(uz)
    tmp = uv - uz
    for j in range(size_v):  # iterate through trials
        Xi = X[:, j*size_u:(j+1)*size_u]
        Xi_ = X_all - Xi
        res_i = Y - Xi.dot(tmp[:, j]) - X_all_uz[:, j]
        grad[size_u + j] = Xi.T.dot(res_i).dot(u)
        grad[size_u + size_v + j] = Xi_.T.dot(res_i).dot(u)
        Z = (Xi * v[j] + Xi_ * z[j])
        grad[:size_u] += Z.T.dot(res_i).ravel()
    grad[:size_u] += u
    return -grad


def f_grad_separate(w, X, Y, drifts, size_u, size_v):
    """TODO: use the vector of drifts"""
    u, v, z = w[:size_u], w[size_u:size_u + size_v], w[size_u + size_v:]
    grad = np.zeros((size_u + 2 * size_v))
    norm = 0
    uv = np.outer(u, v)
    uz = np.outer(u, z)
    X_all_uz = (X[1][0] + X[0][0]).dot(uz)
    tmp = uv - uz
    for j in range(size_v):  # iterate through trials
        Xi = X[0][j]
        Xi_ = X[1][j]
        res_i = Y - Xi.dot(tmp[:, j]) - X_all_uz[:, j]
        Xi_res_ = Xi_.T.dot(res_i)
        Xi_res = Xi.T.dot(res_i)
        grad[size_u + j] = Xi_res.dot(u)
        grad[size_u + size_v + j] = Xi_res_.dot(u)
        assert res_i.ndim == 1
        norm += .5 * linalg.norm(res_i) ** 2
        grad[:size_u] += Xi_res * v[j] + Xi_res_ * z[j] # Z.T.dot(res_i).ravel()
    norm -= .5 * (linalg.norm(u) ** 2)
    grad[:size_u] += u
    return norm, -grad

def rank_one(X, y, n_basis,  w_i=None, drifts=None, callback=None,
    maxiter=500, method='L-BFGS-B', rtol=1e-6,  verbose=0, mode='r1glm',
    hrfs=None, basis=None, bounds=True):
    """
    Estimates a R1-GLM model with a given design matrix.

    This methods solves a problem of the form

        argmin_{hrf, betas} ||y - X vec(hrf betas.T)||^2

    for given y and X

    Parameters
    ----------
    X : array-like, shape (n_scans, n_regressors)
        Design matrix. Note that the number of columns in the design matrix
        must be a multiple of the number of basis elements (n_basis)

    y_i: array-like, shape (n_scans, n_voxels)
        BOLD signal.

    n_basis : int
        Number of basis elements in the HRF.

    w_i : array-like
        initial point.

    method: {'L-BFGS-B', 'TNC'}
        Optimization algorithm.

    bounds: {True, False}
        If True, constraints the estimated HRF (only in the case of 
        basis='2hrf' or basis='3hrf') so that the second and third
        derivative have absolute value < 1.

    Returns
    -------
    U : array
        Estimated HRFs
    V : array
        Estimated activation coefficients
    """
    y = np.array(y)
    if y.ndim > 1:
        n_task = y.shape[1]
    else:
        n_task = 1
        y = y.reshape((y.size, 1))
    if drifts is None:
        drifts = np.zeros((y.shape[0], 1))
    size_v = X.shape[1] // n_basis
    U = []
    V = []
    if not sparse.issparse(X):
        warnings.warn('Matrix X is not in sparse format. This method might be slow')
    if w_i is None:
        if mode == 'r1glm':
            # XXX initialization
            w_i = np.ones((n_basis + size_v + drifts.shape[1], n_task))
        elif mode == 'r1glms':
            w_i = np.random.randn(n_basis + 2 * size_v, n_task)
        else:
            raise NotImplementedError
    w_i = np.array(w_i)
    if mode == 'r1glm':
        assert w_i.shape[0] == n_basis + size_v + drifts.shape[1]
    elif mode == 'r1glms':
        assert w_i.shape[0] == n_basis + 2 * size_v
    assert w_i.shape[1] == n_task

    if mode == 'r1glms':
        E = np.kron(np.ones((size_v, 1)), np.eye(n_basis))
        X_all = sparse.csr_matrix(X.dot(E))
        Xi = []
        Xi_all = []
        for j in range(size_v):  # iterate through trials
            tmp = X[:, j*n_basis:(j+1)*n_basis]
            Xi.append(tmp)
            Xi_all.append(X_all - tmp)
        X = (Xi, Xi_all)
        ofunc = f_grad_separate
    else:
        ofunc = f_grad


    if bounds and basis in ('2hrf', '3hrf'):
        # constrain the derivatives to not go too far
        bounds = [(1, 1)] + [(-1., 1.)] * (n_basis - 1)  + \
            [(None, None)]* (w_i.shape[0] - n_basis)
    else:
        bounds = [(1, 1)]  + \
            [(None, None)]* (w_i.shape[0] - 1)

    if method == 'L-BFGS-B':
        solver = optimize.fmin_l_bfgs_b
    elif method =='TNC':
        solver = optimize.fmin_tnc


    for i in range(n_task):
        args = [X, y[:, i], drifts, n_basis, size_v]
        options = {'maxiter': maxiter}
        if int(verbose) > 1:
            options['disp'] = 5
        else:
            options['disp'] = 0
        kwargs = {}

        out = solver(
            ofunc, w_i[:, i], args=args, bounds=bounds,
            maxiter=maxiter, callback=callback, pgtol=rtol,
            maxfun=30000,
            **kwargs)

        if verbose > 1 and not out[-1]['warnflag'] != 0:
            warnings.warn(out[-1]['task'])
        if int(verbose) > 0:
            if ((i+1) % 500) == 0:
                print('.. completed %s out of %s ..' % (i + 1, n_task))
            if verbose > 1:
                if hasattr(out, 'nit'):
                    print('Number of iterations: %s' % out.nit)
                if hasattr(out, 'fun'):
                    print('Loss function: %s' % out.fun)
        # w_i[:] = out.x.copy()  # use as warm restart
        ui = out[0][:n_basis].ravel()
        vi = out[0][n_basis:n_basis + size_v].ravel()
        U.append(ui)
        V.append(vi)

    U = np.array(U).T
    V = np.array(V).T
    return U, V


def glm(conditions, onsets, TR, Y, drifts=None, basis='3hrf', mode='r1glm',
        hrf_length=20, oversample=5,
        rtol=1e-8, verbose=False, maxiter=500, callback=None,
        method='L-BFGS-B', n_jobs=1, hrfs=None,
        return_design_matrix=False, bounds=True):
    """
    Perform a GLM from BOLD signal, given the conditons, onset,
    TR (repetition time of the scanner) and the BOLD signal.

    This method is able to fir a variety of models, available
    through the `mode` keyword. These are:

        - glm: standard GLM
        - glms: GLM with separate designs
        - r1glm: Rank-1 GLM
        - r1glms: Rank-1 GLM with separate designs

    basis:

        - hrf: single element basis
        - 3hrf: basis with 3 elements
        - fir: basis with hrf_length elements (in multiples of TR)

    **Note** the output parameters need are not normalized.
    Rank-1 models are specified up to a constant
    term between the betas and the HRF. This implies that some
    normalization must be done prior to interpreting the activation
    coefficients. Typically the HRF is normalized to
    have unit amplitude and to correlate positively with a
    reference HRF.


    Parameters
    ----------

    conditions: array-like, shape (n_trials)
        array of conditions

    onsets: array-like, shape (n_trials)
        array of onsets

    TR: float
        Repetition Time, the delay between two succesive
        aquisitions of the same image.

    Y : array-like, shape (n_scans, n_voxels)
        Time-series vector.

    mode: {'r1glm', 'r1glms', 'glms', 'glm'}
        Different GLM models.

    rtol : float
        Relative tolerance for stopping criterion.

    maxiter : int
        maximum number of iterations

    verbose : {0, 1, 2}
        Different levels of verbosity

    n_jobs: int
        Number of CPUs to use. Use -1 to use all available CPUs.

    method: {'L-BFGS-B', 'TNC'}
        Different algorithmic solvers, only used for 'r1*' modes.
        All should yield the same result but their efficiency might vary.
    
    

    Returns
    -------
    U : array
        Estimated HRF. Will be of shape (basis_len, n_voxels) for rank-1
        methods and of (basis_len, n_conditions, n_voxels) for the other
        methods.

    V : array, shape (p, n_voxels)
        Estimated activation coefficients (beta-map).

    dmtx: array,
        Design matrix. Only returned if return_design_matrix=True

    """
    if not mode in ('glm', 'r1glm', 'r1glms', 'glms'):
        raise NotImplementedError
    conditions = np.asarray(conditions)
    onsets = np.asarray(onsets)
    if conditions.size != onsets.size:
        raise ValueError('array conditions and onsets should have the same size')
    Y = np.asarray(Y)
    n_scans = Y.shape[0]
    verbose = int(verbose)
    if verbose > 0:
        print('.. creating design matrix ..')
    if drifts is None:
        drifts = np.ones((n_scans, 1))
    elif mode == 'r1glms':
        print('r1glms does not accept a vector of drifts. drifts will be ignored')


    X_design, Q = create_design_matrix(
        conditions, onsets, TR, n_scans, basis, oversample, hrf_length)
    if verbose > 0:
        print('.. done creating design matrix ..')

    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))
    n_task = Y.shape[1]

    size_u = Q.shape[1]
    size_v = X_design.shape[1] // size_u

    if mode == 'glms':
        U, V = utils.glms_from_glm(
            X_design, Q, n_jobs, False, Y)
    elif mode == 'glm':
        U, V = utils.glm(
            X_design, Q, Y, convolve=False)
    elif mode in ('r1glm', 'r1glms'):
        U = np.zeros((size_u, n_task))
        V = np.zeros((size_v, n_task))
        if verbose > 0:
            print('.. computing initialization ..')
        X_design_canonical, Q_canonical = create_design_matrix(conditions, onsets, TR,
            n_scans, [hrf.spmt], oversample, hrf_length)
        X_design_canonical = np.concatenate(
            (X_design_canonical, drifts), axis=1)
        V_init = linalg.lstsq(X_design_canonical, Y)[0]
        U_init = np.tile(linalg.lstsq(Q, Q_canonical)[0], n_task)
        if mode == 'r1glm':
            W_init = np.concatenate((U_init, V_init))
        else:
            # XXX TODO intercept
            W_init = np.concatenate((U_init, V_init[1:], V_init[1:]))
        if verbose > 0:
            print('.. done initialization ..')

        if n_jobs == -1:
            n_jobs = cpu_count()
        Y_split = np.array_split(Y, n_jobs, axis=1)
        W_init_split = np.array_split(W_init, n_jobs, axis=1)
        X_design = sparse.csr_matrix(X_design)

        out = Parallel(n_jobs=n_jobs)(
            delayed(rank_one)(
                X_design, y_i, size_u, w_i, drifts=drifts, callback=callback, maxiter=maxiter,
                method=method, rtol=rtol, verbose=verbose, mode=mode, hrfs=hrfs, basis=basis,
                bounds=bounds)
            for y_i, w_i in zip(Y_split, W_init_split))

        counter = 0
        for tmp in out:
            u, v = tmp
            u = u.T
            v = v.T
            for i in range(len(u)):
                U[:, counter] = u[i]
                V[:, counter] = v[i]
                counter += 1

        raw_U = U.copy()
        # normalize
    if mode in ('r1glm',) and basis == '3hrf':
        xx = np.linspace(0, hrf_length * TR)
        generated_hrfs = U[0] * hrf.spmt(xx)[:, None] + \
            U[1] * hrf.dspmt(xx)[:, None] + U[2] * hrf.ddspmt(xx)[:, None]
        sign = np.sign(np.dot(generated_hrfs.T, hrf.spmt(xx)))
        norm = np.abs(generated_hrfs).max(0)
        U = U * sign / norm
        V = V * sign * norm
    elif mode in ('r1glm',) and basis == '2hrf':
        xx = np.linspace(0, hrf_length * TR)
        generated_hrfs = U[0] * hrf.spmt(xx)[:, None] + \
            U[1] * hrf.dspmt(xx)[:, None]
        sign = np.sign(np.dot(generated_hrfs.T, hrf.spmt(xx)))
        norm = np.abs(generated_hrfs).max(0)
        U = U * sign / norm
        V = V * sign * norm
    elif mode == 'r1glm' and basis == 'fir':
        xx =  np.arange(0, TR * hrf_length, TR)
        sign = np.sign(np.dot(U.T, hrf.spmt(xx)))
        norm = np.abs(U).max(0)
        U = U * sign / norm
        V = V * sign * norm
    out = [U, V]
    if return_design_matrix:
        out.append(X_design.toarray())
    return out
