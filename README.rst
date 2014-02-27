Hemodynamic Response Function (HRF) estimation from functional MRI data
=======================================================================

This is a Python package that implements several methods for the
joint estimation of HRF and activation patterns (aka beta-map) from
fMRI (BOLD) signal.

If you use this software, please cite at least one of the following papers

"HRF estimation improves sensitivity of fMRI encoding and decoding
models", Fabian Pedregosa, Michael Eickenberg, Bertrand Thirion and
Alexandre Gramfort, `[PDF] <http://hal.inria.fr/docs/00/82/19/46/PDF/paper.pdf>`_
`[URL] <http://hal.inria.fr/hal-00821946/en>`_

XXX bibtex

.. image:: https://raw.github.com/fabianp/hrf_estimation/master/doc/estimation_natural_images.png


Get the code
------------

hrf_estimation is a pure Python package and can be installed through the Python Package Index (PYPI):

.. code:: bash

   pip install -U hrf_estimation

You can also download the source code from the `PYPI website <https://pypi.python.org/pypi/hrf_estimation>`_
or get the latest sources from `github <http://github.com/fabianp/hrf_estimation/>`_


Function reference
------------------

The main function is rank_one, which will compute the estimated HRF and
activation pattern (beta-map) from the BOLD signal.

XXX update

.. code:: python

    def rank_one(X, Q, Y, can_hrf,
                 rtol=1e-8, verbose=False, maxiter=5000, callback=None,
                 method='L-BFGS-B', n_jobs=1, mode='ls', init='LSS',
                 downsample=1):
        """
         multi-target rank one model

             ||y - (X*Q) vec(u v.T)||^2

         Parameters
         ----------
         X : array-like, sparse matrix or LinearOperator, shape (n, n_trials)
             The design matrix

         Q : array-lime, shape (basis_len, n_basis)

         Y : array-lime, shape (n, n_voxels)
             Time-series vector. Several time-series vectors can be given at once,
             however for large system becomes unstable. We do not recommend
             using more than k > 100.

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
         U : array, shape (basis_len, n_voxels)
         V : array, shape (p, n_voxels)
         W : coefficients associated to the drift vectors
         """

The output of this function is XXX


Examples
--------

XXX small examples here!

`This IPython notebook
<http://nbviewer.ipython.org/url/raw.github.com/fabianp/hrf_estimation/master/doc/figures_natural_images.ipynb>`_
contains code that reproduces the figures from the original article.
Development

The newest version can alway be grabbed from the `git repository
<http://github.com/fabianp/hrf_estimation>`_. Feel free to submit
issues or patches.


Authors
-------

`Fabian Pedregosa <http://fa.bianp.net>`_ <f@bianp.net>

Michael Eickenberg <michael.eickenberg@nsup.org>

Thanks to
---------
Yaroslav Halchenko, Bug reports

