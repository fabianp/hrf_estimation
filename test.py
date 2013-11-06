import numpy as np
from scipy import linalg
from hrf_estimation import rank_one

def test_ls():
	"""Test that it defaults to a least squares problem"""
	n_samples, n_features, n_task = 100, 10, 10
	X = np.random.randn(n_samples, n_features)
	W = np.random.randn(n_features, n_task)
	Y = X.dot(W) + 0.5 * np.random.randn(n_samples, n_task)

	u, v = rank_one(X, Y, 1, [1.])
	betas = linalg.lstsq(X, Y)[0]
	res = Y - X.dot(v)
	res = (res * res).sum(0)
	res0 = Y - X.dot(betas)
	res0 = (res0 * res0).sum(0)

	assert np.all((res - res0) / res0 < 1e-3)

if __name__ == '__main__':
	test_ls()