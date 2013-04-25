Hemodynamic Response Function estimation from functional MRI data
=================================================================

This describes a Python package that implements the routines described in the paper

"HRF estimation improves sensitivity of fMRI encoding and decoding
models", Fabian Pedregosa, Michael Eickenberg, Bertrand Thirion and
Alexandre Gramfort (submitted)

Get the code
------------

hrf_estimation is a Python package. It can be installed through the Python Package Index (PYPI):

.. code:: bash

   pip install -U hrf_estimation

You can also download the source code from the PYPI website

Function reference
------------------

The principal function is rank_one

..code::

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


Examples
--------

This IPython notebook contains code that reproduces the figures from the original article.
Development

The newest version can alway be grabbed from the main repository. Feel free to submit issues, modifications or implementations for other languages!.
This package implements the methods from paper XXX

TODO: provide fallback for einsum

Authors
-------

`Fabian Pedregosa <http://fseoane.net>`_ <fabian@fseoane.net>
Michael Eickenberg <michael.eickenberg@nsup.org>
