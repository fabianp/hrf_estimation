import functools
import numpy as np
from numbers import Number
from scipy import linalg, sparse
from scipy.linalg import toeplitz
from joblib import Parallel, delayed, cpu_count

# local import
from . import hrf

def create_design_matrix(conditions, onsets, TR, n_scans, basis='dhrf',
                         oversample=100, hrf_length=30):
    """
    Parameters
    ----------
    conditions:
    onset:
    """
    if basis == 'dhrf':
        basis = [hrf.spmt, hrf.dspmt, hrf.ddspmt]
    elif basis == 'hrf':
        basis = [hrf.spmt]
    elif basis == 'fir':
        basis = []
        for i in range(int(20. / TR)):
            tmp = functools.partial(hrf.fir, i=i, TR=TR)
            # import pylab as pl
            # xx = np.linspace(0, 20)
            # pl.plot(xx, tmp(xx)); pl.show()
            basis.append(tmp)

    resolution = TR / float(oversample)
    conditions = np.asarray(conditions)
    onsets = np.asarray(onsets, dtype=np.float)
    unique_conditions = np.unique(conditions)
    design_matrix_cols = []
    B = []
    for b in basis:
        tmp_basis = b(np.arange(0, hrf_length, resolution))
        # import pylab as pl
        # pl.plot(np.arange(0, hrf_length, resolution), tmp_basis)
        # pl.show()
        # import ipdb; ipdb.set_trace()
        B.append(tmp_basis)
    for c in unique_conditions:
        tmp = np.zeros(int(n_scans * TR/resolution))
        onset_c = onsets[conditions == c]
        idx = np.round(onset_c / resolution).astype(np.int)
        tmp[idx] = 1.
        for b in B:
            col = np.convolve(b, tmp, mode='full')[:tmp.size]
            design_matrix_cols.append(col[::oversample])

    design_matrix = np.array(design_matrix_cols).T
    assert design_matrix.shape[0] == n_scans
    return design_matrix, np.asarray(B).T[::oversample] # XXX hrf_length

def convolution_matrix(kernel=[1], signal_length=15):
    """
    Creates a matrix representing the convolution operation with
    the given kernel, padded to the correct signal length
    """

    return toeplitz(list(kernel) + [0] * (signal_length - 1),
                    [0] * signal_length)[:-len(kernel) + 1]


def convolve_events(event_matrix, hrf_basis=20, sparse_output=False):
    """
    Takes a design matrix containing events and convolves it with hrf basis
    If hrf_basis is a number, it will use a cartesian basis of that size
    """

    if isinstance(hrf_basis, Number):
        hrf_basis = np.eye(hrf_basis)

    fir_matrix = np.zeros(list(event_matrix.shape) + [hrf_basis.shape[1]])

    for i in range(hrf_basis.shape[1]):
        conv_mat = convolution_matrix(hrf_basis[:, i], len(event_matrix))
        if conv_mat.size:
            fir_matrix[:, :, i] = conv_mat.dot(event_matrix)
        else:
            fir_matrix[:, :, i] = event_matrix

    out = fir_matrix.reshape(len(event_matrix), -1)
    if sparse_output:
        return sparse.csr_matrix(out)
    return out


def classic_to_obo(classic_design, fir_length=1):
    """
    Will convert a classic or fir design to the one by one setting.
    Returns one matrix per event, containing event related regressors
    on the left and sum of remaining regressors on the right.
    """

    event_regressors = classic_design.reshape(
        len(classic_design), -1, fir_length)

    regressor_sum = event_regressors.sum(axis=1)

    remaining_regressors = (regressor_sum[:, np.newaxis, :] -
                            event_regressors)
    together = np.concatenate([event_regressors,
                               remaining_regressors], axis=2)

    return together.transpose(1, 0, 2)


def glm(event_matrix, Q, voxels, hrf_function=None, downsample=1,
        convolve=True):
    """
    Perform a GLM from an event matrix
    and return estimated HRFs and associated coefficients
    Q: basis
    """
    Q = np.asarray(Q)
    if Q.ndim == 1:
        Q = Q[:, None]
    if hrf_function is None:
        hrf_function = Q[:, 0]
    if convolve:
        glm_design = convolve_events(event_matrix, Q)[::downsample]
    else:
        glm_design = event_matrix
    n_basis = Q.shape[1]
    n_trials = glm_design.shape[1] / n_basis
    n_voxels = voxels.shape[-1]
    full_betas = linalg.lstsq(glm_design, voxels)[0]
    full_betas = full_betas.reshape(n_basis, n_trials, n_voxels, order='F')
    hrfs = full_betas.T.dot(Q.T)
    sign = np.sign((hrfs * hrf_function).sum(-1))
    hrfs = hrfs * sign[..., None]
    norm = hrfs.max(-1)
    hrfs /= norm[..., None]
    betas = norm * sign
    return hrfs.T, betas.T


def _separate_innerloop(glms_design, n_basis, voxels):
    betas = np.empty((n_basis, glms_design.shape[0], voxels.shape[-1]))
    w = np.empty_like(betas)
    for i in range(glms_design.shape[0]):
        tmp = linalg.lstsq(glms_design[i], voxels)[0]
        betas[:, i, :] = tmp[:n_basis]
        w[:, i, :] = tmp[n_basis:]
    return betas, w


def glms_from_glm(glm_design, Q, ref_hrf, n_jobs, return_w, voxels):
    """
    Performs a GLM-separate from a GLM design matrix as input

    Needs a numpy array (no sparse matrix) as input
    """
    n_basis = Q.shape[1]
    glms_design = classic_to_obo(glm_design, n_basis)
    if n_jobs == -1:
        n_jobs = cpu_count()
    glms_split = np.array_split(glms_design, n_jobs, axis=0)
    out = Parallel(n_jobs=n_jobs)(
        delayed(_separate_innerloop)(glms_i, n_basis, voxels)
        for glms_i in glms_split)
    betas = []
    w = []
    for o in out:
        betas.append(o[0])
        w.append(o[1])
    full_betas = np.concatenate(betas, axis=1)
    full_w = np.concatenate(w, axis=1)
    hrfs = full_betas.T.dot(Q.T)
    sign = np.sign((hrfs * ref_hrf).sum(-1))
    hrfs = hrfs * sign[..., None]
    # norm = hrfs.max(-1)
    norm = np.sqrt((hrfs * hrfs).sum(-1))
    hrfs /= norm[..., None]
    betas = norm * sign
    if return_w:
        hrfs_w = full_w.T.dot(Q.T)
        norm_w = np.sqrt((hrfs_w * hrfs_w).sum(-1))
        sign_w = np.sign((hrfs_w * ref_hrf).sum(-1))
        hrfs_w = hrfs_w * sign[..., None] / norm_w[..., None]
        betas_w = norm_w * sign_w
        return hrfs.T, betas.T, betas_w.T
    return hrfs.T, betas.T


def glm_separate(event_matrix, Q, voxels, hrf_function, n_jobs=1,
                 return_w=False, downsample=1):
    """
    Perform a GLM with separate designs from an event matrix
    """
    if sparse.issparse(event_matrix):
        event_matrix = event_matrix.toarray()
    n_basis = Q.shape[1]
    n_conditions = event_matrix.shape[1]
    glm_design = convolve_events(event_matrix, Q)[::downsample]
    return glms_from_glm(glm_design, Q, hrf_function, n_jobs, return_w, voxels)
