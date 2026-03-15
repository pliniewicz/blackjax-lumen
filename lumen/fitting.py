"""
SED data I/O and log-likelihood.

Data structures are plain Python (not JAX-traced) since they represent
fixed observations.  The log-likelihood *is* JAX-traceable w.r.t. model
parameters, enabling gradient-based optimization and HMC/NUTS sampling.
"""

import jax
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

from .sed import observed_sed
from .cosmology import Cosmology


# ------------------------------------------------------------------ #
#  Data containers                                                    #
# ------------------------------------------------------------------ #

@dataclass
class SEDDataPoint:
    nu: float           # frequency [Hz]
    nuFnu: float        # νFν [erg/s/cm²]
    nuFnu_err: float    # error on νFν
    is_upper: bool = False


@dataclass
class SEDData:
    points: List[SEDDataPoint]

    @property
    def frequencies(self):
        return jnp.array([p.nu for p in self.points])

    @property
    def fluxes(self):
        return jnp.array([p.nuFnu for p in self.points])

    @property
    def errors(self):
        return jnp.array([p.nuFnu_err for p in self.points])

    @property
    def upper_limit_mask(self):
        return jnp.array([p.is_upper for p in self.points])

    @property
    def has_upper_limits(self):
        return any(p.is_upper for p in self.points)

    @property
    def log_fluxes(self):
        """log10(νFν) for each data point."""
        return jnp.log10(jnp.array([p.nuFnu for p in self.points]))

    @property
    def log_errors(self):
        """Uncertainty on log10(νFν), propagated from linear errors.

        σ_log10 = σ_lin / (flux · ln(10))
        """
        return jnp.array([
            p.nuFnu_err / (p.nuFnu * np.log(10)) for p in self.points
        ])

    def detections(self):
        return SEDData([p for p in self.points if not p.is_upper])

    def upper_limits(self):
        return SEDData([p for p in self.points if p.is_upper])

    def __len__(self):
        return len(self.points)


# ------------------------------------------------------------------ #
#  Loading                                                            #
# ------------------------------------------------------------------ #

def load_sed(filename: str, delimiter=',', skip_header: int = 1,
             log_data: bool = False) -> SEDData:
    """
    Load SED from file.  Columns: ν, νFν, νFν_err [, is_upper].

    When log_data=True, input columns are log10 values and errors
    are propagated: σ_lin = 10^val · ln(10) · σ_log.
    """
    raw = np.loadtxt(filename, delimiter=delimiter, skiprows=skip_header)
    n, ncols = raw.shape

    points = []
    for i in range(n):
        is_upper = bool(raw[i, 3] == 1) if ncols >= 4 else False

        if log_data:
            nu = 10.0 ** raw[i, 0]
            nuFnu = 10.0 ** raw[i, 1]
            nuFnu_err = nuFnu * np.log(10) * raw[i, 2] if ncols >= 3 else 0.1 * nuFnu
            if nuFnu_err == 0.0:
                nuFnu_err = 0.1 * nuFnu
        else:
            nu = raw[i, 0]
            nuFnu = raw[i, 1]
            nuFnu_err = raw[i, 2] if ncols >= 3 else 0.1 * nuFnu
            if nuFnu_err == 0.0:
                nuFnu_err = 0.1 * nuFnu

        points.append(SEDDataPoint(nu, nuFnu, nuFnu_err, is_upper))

    return SEDData(points)


def load_sed_Fnu(filename: str, unit: str = 'mJy',
                 delimiter=',', skip_header: int = 1) -> SEDData:
    """
    Load SED with Fν columns.  Columns: ν [Hz], Fν, ΔFν [, is_upper].
    Converts to νFν internally.
    """
    from .cosmology import Fnu_to_nuFnu

    raw = np.loadtxt(filename, delimiter=delimiter, skiprows=skip_header)
    n, ncols = raw.shape

    points = []
    for i in range(n):
        nu = raw[i, 0]
        Fnu = raw[i, 1]
        dFnu = raw[i, 2] if ncols >= 3 else 0.1 * Fnu
        if dFnu == 0.0:
            dFnu = 0.1 * Fnu
        is_upper = bool(raw[i, 3] == 1) if ncols >= 4 else False

        nuFnu = float(Fnu_to_nuFnu(nu, Fnu, unit))
        nuFnu_err = float(Fnu_to_nuFnu(nu, dFnu, unit))
        points.append(SEDDataPoint(nu, nuFnu, nuFnu_err, is_upper))

    return SEDData(points)


# ------------------------------------------------------------------ #
#  Log-likelihood                                                     #
# ------------------------------------------------------------------ #

def _build_loglik_fn(nu_obs, log_flux_obs, log_flux_err, is_upper,
                     cosmo, nx, ngamma, use_kn):
    """Build a pure, JIT-compiled log-likelihood closure over fixed data.

    The returned function has signature ``loglik(params) -> float``
    and is JAX-traceable w.r.t. params (suitable for jax.grad, HMC, etc.).
    """
    has_ul = jnp.any(is_upper)

    @partial(jax.jit)
    def _loglik(params):
        nuFnu_model = observed_sed(nu_obs, params, cosmo, nx, ngamma, use_kn)
        log_flux_model = jnp.log10(jnp.maximum(nuFnu_model, 1e-300))

        # Detections: standard Gaussian in log10(νFν)
        residuals = (log_flux_obs - log_flux_model) / log_flux_err
        ll_det = -0.5 * residuals ** 2

        # Upper limits: log Φ((UL - model) / σ)
        # model << UL → ~0 penalty; model >> UL → large negative penalty
        ll_ul = jax.scipy.stats.norm.logcdf(
            (log_flux_obs - log_flux_model) / log_flux_err
        )

        return jnp.sum(jnp.where(is_upper, ll_ul, ll_det))

    return _loglik


def log_likelihood(params, *, data: SEDData, cosmo: Cosmology,
                   nx: int = 96, ngamma: int = 128, use_kn: bool = True):
    """
    Gaussian log-likelihood in log10(νFν) space.

    Accepts an SEDData object directly; linear-to-log conversion is
    handled internally.  Upper limits (if present) are automatically
    handled via the normal log-CDF.

    Parameters
    ----------
    params : JetParams
        Model parameters (JAX-traced).
    data : SEDData
        Observed SED (frequencies, fluxes, errors, upper-limit flags).
    cosmo : Cosmology
        Cosmological parameters.
    nx, ngamma : int
        Quadrature orders.
    use_kn : bool
        Include Klein-Nishina corrections.

    Returns
    -------
    float
        log p(data | params), up to a constant.
    """
    loglik_fn = _build_loglik_fn(
        data.frequencies, data.log_fluxes, data.log_errors,
        data.upper_limit_mask, cosmo, nx, ngamma, use_kn,
    )
    return loglik_fn(params)


def make_log_likelihood(data: SEDData, cosmo: Cosmology,
                        nx: int = 96, ngamma: int = 128,
                        use_kn: bool = True):
    """
    Return a compiled ``loglik(params) -> float`` closure.

    Use this when evaluating the likelihood many times (e.g. in a
    sampling loop) to avoid rebuilding the closure on every call.

    Example
    -------
    >>> loglik = lumen.make_log_likelihood(data, cosmo=Planck18)
    >>> loglik(params)            # fast, JIT-compiled
    >>> jax.grad(loglik)(params)  # autodiff just works
    """
    return _build_loglik_fn(
        data.frequencies, data.log_fluxes, data.log_errors,
        data.upper_limit_mask, cosmo, nx, ngamma, use_kn,
    )

