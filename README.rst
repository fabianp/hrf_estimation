Hemodynamic Response Function estimation from functional MRI data
=================================================================

This describes a Python package that implements the routines described in the paper

"HRF estimation improves sensitivity of fMRI encoding and decoding
models", Fabian Pedregosa, Michael Eickenberg, Bertrand Thirion and
Alexandre Gramfort (submitted)

.. image:: https://raw.github.com/fabianp/hrf_estimation/master/doc/estimation_natural_images.png

Get the code
------------

hrf_estimation is a Python package. It can be installed through the Python Package Index (PYPI):

.. code:: bash

   pip install -U hrf_estimation

You can also download the source code from the `PYPI website <https://pypi.python.org/pypi/hrf_estimation>`_

Function reference
------------------

The principal function is rank_one

.. code:: python

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


Examples
--------

`This IPython notebook
<http://nbviewer.ipython.org/url/raw.github.com/fabianp/hrf_estimation/master/doc/figures_prni2013_gallant.ipynb>`_
contains code that reproduces the figures from the original article.
Development

The newest version can alway be grabbed from the `git repository
<http://github.com/fabianp/hrf_estimation>`_. Feel free to submit
patches, issues or implementations for other languages!.

TODO: provide fallback for einsum

Authors
-------

`Fabian Pedregosa <http://fseoane.net>`_ <f@fabianp.net>
Michael Eickenberg <michael.eickenberg@nsup.org>
