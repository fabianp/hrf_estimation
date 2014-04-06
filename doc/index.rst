.. hrf_estimation documentation master file, created by
   sphinx-quickstart on Thu Feb 27 09:10:10 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Main functions
==============

The main function in the package is called `hrf_estimation.glm` and will extract the
HRF and activation coefficients from BOLD signal. 

This function takes as input a vector of conditions, a vector of onsets, the TR (float) and the matrix of BOLD measurements (of size n_timecourse * n_voxels). Its signature is

.. autofunction:: hrf_estimation.glm


If you have already formed the design matrix and only want to estimate a R1-GLM model from that design matrix and a matrix fo BOLD measurements, then the function ``rank_one`` can be used. Its signature is

.. autofunction:: hrf_estimation.rank_one

Examples
========

Coming soon


.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

