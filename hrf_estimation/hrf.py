# This module was copied from nipy (with removed dependency from sympy)
""" This module provides definitions of various hemodynamic response
functions (hrf).

In particular, it provides Gary Glover's canonical HRF, AFNI's default
HRF, and a spectral HRF.

The Glover HRF is based on:

@article{glover1999deconvolution,
  title={{Deconvolution of impulse response in event-related BOLD fMRI}},
  author={Glover, G.H.},
  journal={NeuroImage},
  volume={9},
  number={4},
  pages={416--429},
  year={1999},
  publisher={Orlando, FL: Academic Press, c1992-}
}

This parametrization is from fmristat:

http://www.math.mcgill.ca/keith/fmristat/

fmristat models the HRF as the difference of two gamma functions, ``g1``
and ``g2``, each defined by the timing of the gamma function peaks
(``pk1, pk2``) and the FWHMs (``width1, width2``):

   raw_hrf = g1(pk1, width1) - a2 * g2(pk2, width2)

where ``a2`` is the scale factor for the ``g2`` gamma function.  The
actual hrf is the raw hrf set to have an integral of 1.

fmristat used ``pk1, width1, pk2, width2, a2 = (5.4 5.2 10.8 7.35
0.35)``.  These are parameters to match Glover's 1 second duration
auditory stimulus curves.  Glover wrote these as:

   y(t) = c1 * t**n1 * exp(t/t1) - a2 * c2 * t**n2 * exp(t/t2)

with ``n1, t1, n2, t2, a2 = (6.0, 0.9, 12, 0.9, 0.35)``.  The difference
between Glover's expression and ours is because we (and fmristat) use
the peak location and width to characterize the function rather than
``n1, t1``.  The values we use are equivalent.  Specifically, in our
formulation:

>>> n1, t1, c1 = gamma_params(5.4, 5.2)
>>> np.allclose((n1-1, t1), (6.0, 0.9), rtol=0.02)
True
>>> n2, t2, c2 = gamma_params(10.8, 7.35)
>>> np.allclose((n2-1, t2), (12.0, 0.9), rtol=0.02)
True
"""
from __future__ import division

from functools import partial

import numpy as np

import scipy.stats as sps



# SPMs HRF
def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio = 6,
                   normalize=True,
                  ):
    """ SPM HRF function from sum of two gamma PDFs

    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.

    The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
    `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
    and dispersion `under_disp`, and divided by the `p_u_ratio`).

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF
    peak_delay : float, optional
        delay of peak
    peak_disp : float, optional
        width (dispersion) of peak
    under_delay : float, optional
        delay of undershoot
    under_disp : float, optional
        width (dispersion) of undershoot
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning.  SPM does this
        by default.

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`

    Notes
    -----
    See ``spm_hrf.m`` in the SPM distribution.
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.max(hrf)

def _get_num_int(lf, dt=0.02, t=50):
    # numerical integral of numerical function
    tt = np.arange(dt,t+dt,dt)
    return lf(tt).sum() * dt

_spm_can_int = _get_num_int(partial(spm_hrf_compat, normalize=True))


def spmt(t):
    """ SPM canonical HRF, HRF values for time values `t`

    This is the canonical HRF function as used in SPM. It
    has the following defaults:
                                                defaults
                                                (seconds)
    delay of response (relative to onset)         6
    delay of undershoot (relative to onset)      16
    dispersion of response                        1
    dispersion of undershoot                      1
    ratio of response to undershoot               6
    onset (seconds)                               0
    length of kernel (seconds)                   32
    """
    return spm_hrf_compat(t, normalize=True)


def dspmt(t):
    """ SPM canonical HRF derivative, HRF derivative values for time values `t`

    This is the canonical HRF derivative function as used in SPM.

    It is the numerical difference of the HRF sampled at time `t` minus the
    values sampled at time `t` -1
    """
    t = np.asarray(t)
    return spmt(t) - spmt(t - 1)


_spm_dd_func = partial(spm_hrf_compat, normalize=True, peak_disp=1.01)

def ddspmt(t):
    """ SPM canonical HRF dispersion derivative, values for time values `t`

    This is the canonical HRF dispersion derivative function as used in SPM.

    It is the numerical difference between the HRF sampled at time `t`, and
    values at `t` for another HRF shape with a small change in the peak
    dispersion parameter (``peak_disp`` in func:`spm_hrf_compat`).
    """
    return (spmt(t) - _spm_dd_func(t)) / 0.01

def fir(t, i=0, TR=1.):
    """
    The FIR basis
    formed by the canonical vectors with a `resolution` of 1 TR

    Parameters
    ----------
    t: TODO
    i: TODO
    TR: TODO
    """
    return np.bitwise_and(i * TR <= t, (i+1) * TR > t).astype(np.float)

