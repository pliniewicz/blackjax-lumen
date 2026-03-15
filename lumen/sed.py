"""
Observed SED: rest-frame luminosities → observer-frame νFν.

Includes synchrotron, IC/CMB, and total (synch + IC).
"""

import jax
import jax.numpy as jnp
from functools import partial

from .cosmology import luminosity_distance_cm, Cosmology
from .synchrotron import synchrotron_spectrum
from .ic import ic_spectrum


@partial(jax.jit, static_argnums=(3, 4))
def observed_synchrotron(nu_array, params, cosmo: Cosmology,
                         nx: int = 96, ngamma: int = 128):
    """Observed synchrotron νFν [erg/s/cm²]."""
    nuLnu = synchrotron_spectrum(nu_array, params, nx, ngamma)
    d_L = luminosity_distance_cm(params.z, cosmo)
    return nuLnu / (4.0 * jnp.pi * d_L ** 2)


@partial(jax.jit, static_argnums=(3, 4, 5))
def observed_ic(nu_array, params, cosmo: Cosmology,
                nx: int = 96, ngamma: int = 128, use_kn: bool = True):
    """Observed IC/CMB νFν [erg/s/cm²]."""
    nuLnu = ic_spectrum(nu_array, params, nx, ngamma, use_kn)
    d_L = luminosity_distance_cm(params.z, cosmo)
    return nuLnu / (4.0 * jnp.pi * d_L ** 2)


@partial(jax.jit, static_argnums=(3, 4, 5))
def observed_sed(nu_array, params, cosmo: Cosmology,
                 nx: int = 96, ngamma: int = 128, use_kn: bool = True):
    """Observed total (synchrotron + IC/CMB) νFν [erg/s/cm²]."""
    nuLnu_syn = synchrotron_spectrum(nu_array, params, nx, ngamma)
    nuLnu_ic = ic_spectrum(nu_array, params, nx, ngamma, use_kn)
    d_L = luminosity_distance_cm(params.z, cosmo)
    return (nuLnu_syn + nuLnu_ic) / (4.0 * jnp.pi * d_L ** 2)
