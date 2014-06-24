import numpy as np
from scipy import linalg, optimize
import hrf_estimation as he


def test_grad():
    n_samples, n_features = 12, 20
    size_u, size_v = 2, 10
    np.random.seed(0)
    w = np.random.randn(size_u + size_v + 1)
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples)
    func = lambda x: he.f_grad(x, X, Y, size_u, size_v)[0]
    grad = lambda x: he.f_grad(x, X, Y, size_u, size_v)[1]
    assert optimize.check_grad(func, grad, w) < .1
    #
    # X = np.random.randn(n_samples, n_features)
    # X_all = np.random.randn(n_samples, size_u)
    # w = np.random.randn(size_u + 2 * size_v)
    # func = lambda x: he.f_grad_separate(x, X, Y, size_u, size_v)[0]
    # grad = lambda x: he.f_grad_separate(x, X, Y, size_u, size_v)[1]
    # assert optimize.check_grad(func, grad, w) < .1


# def test_ls():
#     """Test that it defaults to a least squares problem"""
#     conditions = [1, 2, 1, 2, 3]
#     onsets = range(0, 50, 10)
#     Y = np.random.randn(50, 10)
#     TR = 1.
#     glm_design, Q = he.utils.create_design_matrix(
#         conditions, onsets, TR, Y.shape[0],
#         basis='hrf')

#     betas = linalg.lstsq(glm_design, Y)[0]

#     hrfs1, betas1 = he.glm(conditions, onsets, TR, Y, basis='hrf')
#     assert np.allclose(betas * Q.max(0), betas1, rtol=1e-3)

    # XXX TODO rank1 separate model

# def test_convergence():
#     n_samples, n_features, n_task = 100, 10, 100
#     Q = np.eye(2)
#     X = np.random.randn(n_samples, n_features / 2.)
#     X_conv = he.utils.convolve_events(X, Q)
#     W = np.random.randn(n_features, n_task)
#     Y = X_conv.dot(W) + 0.5 * np.random.randn(n_samples, n_task)
#     U, V = he.rank_one(X, Q, Y, [1., 1.], rtol=1e-32)
#     for i in range(n_task):
#         w = np.concatenate((U[:, i], V[:, i]))
#         assert np.allclose(he.fprime(w, X_conv, Y[:, i], 2, 5)[2:],
#             np.zeros_like(w[2:]), atol=1e-3)

def test_glm():
    pass

if __name__ == '__main__':
    test_grad()
    # test_ls()