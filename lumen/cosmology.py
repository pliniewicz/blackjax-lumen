"""
Cosmological calculations following Hogg (1999), arXiv:astro-ph/9905116.

Mirrors cosmology.jl.  All distance functions are JAX-traceable w.r.t. z.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple
from functools import partial

from .constants import c_km_s, Mpc_cm
from .integration import get_gl, map_to_interval, gl_quad


class Cosmology(NamedTuple):
    H0: float       # Hubble constant [km/s/Mpc]
    Omega_M: float   # Matter density
    Omega_L: float   # Dark energy density
    Omega_rad: float  # Radiation density
    Omega_k: float   # Curvature (derived)


def make_cosmology(H0, Omega_M, Omega_L, Omega_rad=0.0):
    Omega_k = 1.0 - Omega_M - Omega_L - Omega_rad
    return Cosmology(H0, Omega_M, Omega_L, Omega_rad, Omega_k)


# Standard cosmologies
Planck18 = make_cosmology(67.4, 0.315, 0.685, 9.0e-5)
WMAP9 = make_cosmology(69.3, 0.287, 0.713, 8.5e-5)


def E_z(z, cosmo: Cosmology):
    """Dimensionless Hubble parameter E(z) = H(z)/H0."""
    zp1 = 1.0 + z
    return jnp.sqrt(
        cosmo.Omega_M * zp1**3
        + cosmo.Omega_k * zp1**2
        + cosmo.Omega_L
        + cosmo.Omega_rad * zp1**4
    )


def hubble_distance(cosmo: Cosmology):
    """d_H = c / H0 [Mpc]."""
    return c_km_s / cosmo.H0


def comoving_distance(z, cosmo: Cosmology, n: int = 128):
    """Comoving distance d_C [Mpc]."""
    d_H = hubble_distance(cosmo)
    nodes, weights = get_gl(n)
    zp_nodes, scale = map_to_interval(nodes, 0.0, z)
    integrand_vals = 1.0 / E_z(zp_nodes, cosmo)
    return d_H * gl_quad(integrand_vals, weights, scale)


def transverse_comoving_distance(z, cosmo: Cosmology, n: int = 128):
    """Transverse comoving distance d_M [Mpc]."""
    d_H = hubble_distance(cosmo)
    d_C = comoving_distance(z, cosmo, n)
    Ok = cosmo.Omega_k

    # Use jnp.where for JAX-traceable branching
    sqrt_Ok = jnp.sqrt(jnp.abs(Ok) + 1e-30)

    d_pos = d_H / sqrt_Ok * jnp.sinh(sqrt_Ok * d_C / d_H)
    d_neg = d_H / sqrt_Ok * jnp.sin(sqrt_Ok * d_C / d_H)
    d_flat = d_C

    return jnp.where(
        Ok > 1e-10, d_pos,
        jnp.where(Ok < -1e-10, d_neg, d_flat)
    )


def angular_diameter_distance(z, cosmo: Cosmology, n: int = 128):
    """Angular diameter distance [Mpc]."""
    return transverse_comoving_distance(z, cosmo, n) / (1.0 + z)


def luminosity_distance(z, cosmo: Cosmology, n: int = 128):
    """Luminosity distance [Mpc]."""
    return transverse_comoving_distance(z, cosmo, n) * (1.0 + z)


def luminosity_distance_cm(z, cosmo: Cosmology, n: int = 128):
    """Luminosity distance [cm]."""
    return luminosity_distance(z, cosmo, n) * Mpc_cm


def kpc_per_arcsec(z, cosmo: Cosmology, n: int = 128):
    """Angular scale [kpc/arcsec]."""
    return angular_diameter_distance(z, cosmo, n) * 1000.0 / 206265.0


# ------------------------------------------------------------------ #
#  Flux ↔ Luminosity conversions                                      #
# ------------------------------------------------------------------ #

def nuFnu_to_nuLnu(nuFnu, z, cosmo, n=128):
    d_L = luminosity_distance_cm(z, cosmo, n)
    return 4.0 * jnp.pi * d_L**2 * nuFnu


def nuLnu_to_nuFnu(nuLnu, z, cosmo, n=128):
    d_L = luminosity_distance_cm(z, cosmo, n)
    return nuLnu / (4.0 * jnp.pi * d_L**2)


def Fnu_to_nuFnu(nu, Fnu, unit='mJy'):
    """Convert Fν [in given unit] to νFν [erg/s/cm²]."""
    scale = {
        'Jy': 1e-23, 'mJy': 1e-26, 'uJy': 1e-29,
        'μJy': 1e-29, 'nJy': 1e-32, 'cgs': 1.0,
    }
    if unit not in scale:
        raise ValueError(f"Unknown unit: {unit}. Use Jy/mJy/uJy/nJy/cgs.")
    return nu * Fnu * scale[unit]
