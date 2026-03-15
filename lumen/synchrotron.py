"""
Synchrotron emission — fully vectorized JAX implementation.

Uses the Aharonian, Kelner & Prosekin (2010) approximation for the
synchrotron kernel, avoiding modified Bessel functions that are
unavailable in jax.scipy.

Reference:
    Aharonian, Kelner & Prosekin, Phys. Rev. D 82, 043002 (2010)
    Crusius & Schlickeiser (1986) — original R(x) formulation

The double integral over jet radius x ∈ [0,1] and electron Lorentz
factor γ is evaluated on a tensorized Gauss-Legendre grid and
contracted with weights — no Python loops, fully JIT-able and
autodiff-compatible.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .constants import me, c, q, h
from .integration import get_gl, map_to_interval, gl_quad
from .profiles import (
    Gamma_profile, b_profile, Doppler_factor, B1, PB1,
)
from .electrons import electron_distribution, electron_norm, K_e


# ------------------------------------------------------------------ #
#  Synchrotron kernel: Aharonian approximation                        #
# ------------------------------------------------------------------ #

def synchR(x):
    """
    Synchrotron kernel R(x) — Aharonian et al. (2010) approximation.

    Smooth, positive, and well-behaved for x ∈ (0, ~500).
    Replaces the Crusius & Schlickeiser (1986) Bessel-function form
    used in the Julia code.
    """
    x_safe = jnp.clip(x, 1e-30, 500.0)

    x_13 = jnp.power(x_safe, 1.0 / 3.0)
    x_23 = x_13 * x_13
    x_43 = x_23 * x_23

    term1 = 1.808 * x_13 / jnp.sqrt(1.0 + 3.4 * x_23)
    term2 = (1.0 + 2.210 * x_23 + 0.347 * x_43) / \
            (1.0 + 1.353 * x_23 + 0.217 * x_43)

    return term1 * term2 * jnp.exp(-x_safe)


# ------------------------------------------------------------------ #
#  Precomputation of position- and energy-dependent quantities        #
# ------------------------------------------------------------------ #

def _precompute_x_quantities(x_nodes, params, pb1, enorm):
    """
    Vectorized precomputation of x-dependent jet quantities.

    Returns arrays of shape (nx,) for B(x), δ(x), K_e(x), x.
    """
    model = int(params.model)

    def _single_x(x):
        bx = b_profile(x, model)
        Gx = Gamma_profile(x, params.G0)
        Bx = bx * jnp.sqrt(8.0 * jnp.pi * pb1) / Gx   # b(x) * B1 / Γ(x)
        dx = Doppler_factor(x, params.G0, params.theta)
        Kx = K_e(x, params, pb1, enorm)
        return Bx, dx, Kx

    # vmap over the x-node array
    B_xs, delta_xs, K_xs = jax.vmap(_single_x)(x_nodes)
    return B_xs, delta_xs, K_xs


def _precompute_gamma_quantities(log_gamma_nodes, params):
    """
    Vectorized precomputation of γ-dependent quantities.

    Returns arrays of shape (ngamma,) for γ and f_e(γ).
    """
    gammas = jnp.exp(log_gamma_nodes)
    f_es = electron_distribution(
        gammas, params.p, params.gamma_min, params.gamma_max
    )
    return gammas, f_es


# ------------------------------------------------------------------ #
#  Core spectrum computation                                          #
# ------------------------------------------------------------------ #

@partial(jax.jit, static_argnums=(2, 3))
def synchrotron_spectrum(nu_array, params, nx: int = 96, ngamma: int = 128):
    """
    Compute synchrotron νLν [erg/s] at each frequency in nu_array.

    Parameters
    ----------
    nu_array : jax array, shape (N,)
        Observer-frame frequencies [Hz].
    params : JetParams
        Jet parameters (NamedTuple, pytree).
    nx : int
        Quadrature order in radial direction.
    ngamma : int
        Quadrature order in electron energy.

    Returns
    -------
    nuLnu : jax array, shape (N,)
        Synchrotron luminosity νLν [erg/s].
    """
    # --- Shared precomputations (independent of ν) ---
    pb1 = PB1(params)
    enorm = electron_norm(params)
    zp1 = params.z + 1.0

    # x grid: [0, 1]
    nodes_x, weights_x = get_gl(nx)
    x_nodes, sx = map_to_interval(nodes_x, 0.0, 1.0)

    # log(γ) grid: integrate over [γ_min, γ_max] matching the power-law support.
    # If exponential cutoffs are re-enabled, extend to [γ_min/2, 3·γ_max].
    log_ga = jnp.log(params.gamma_min)
    log_gb = jnp.log(params.gamma_max)
    nodes_g, weights_g = get_gl(ngamma)
    log_gamma_nodes, sg = map_to_interval(nodes_g, log_ga, log_gb)

    # Precompute x- and γ-dependent quantities: shapes (nx,) and (ng,)
    B_xs, delta_xs, K_xs = _precompute_x_quantities(
        x_nodes, params, pb1, enorm
    )
    gammas, f_es = _precompute_gamma_quantities(log_gamma_nodes, params)

    # --- Build the integrand on the (ν, x, γ) grid ---
    # We want: for each ν, double-integrate over x and γ.
    #
    # Strategy: vectorize over ν with vmap; the inner double sum
    # is a contraction of a 2-D (nx, ng) integrand matrix.

    def _integrand_single_nu(nu):
        """Compute the double integral for one frequency."""
        # R argument: shape (nx, ng) via broadcasting
        # Rarg = 4π me c (1+z) ν / (3q δ(x) B(x) γ²)
        #   B_xs: (nx,)  delta_xs: (nx,)  gammas: (ng,)
        Rarg = (4.0 * jnp.pi * me * c * zp1 * nu) / (
            3.0 * q * delta_xs[:, None] * B_xs[:, None] * gammas[None, :] ** 2
        )
        R_vals = synchR(Rarg)  # (nx, ng)

        # Full integrand on the (x, γ) grid
        # x · B(x) · δ(x)² · K_e(x) · f_e(γ) · R(x,γ) · γ  [Jacobian]
        integrand = (
            x_nodes[:, None]
            * B_xs[:, None]
            * delta_xs[:, None] ** 2
            * K_xs[:, None]
            * f_es[None, :]
            * R_vals
            * gammas[None, :]  # Jacobian from log-γ transform
        )

        # Inner sum (γ), then outer sum (x)
        inner = jnp.dot(integrand, weights_g) * sg   # (nx,)
        outer = jnp.dot(weights_x, inner) * sx        # scalar

        return outer

    # vmap over all frequencies
    double_integrals = jax.vmap(_integrand_single_nu)(nu_array)

    # Prefactor: 2π √3 q³ Rj² l (1+z) ν / (me c²)
    prefactor_base = (
        2.0 * jnp.pi * jnp.sqrt(3.0) * q**3
        * params.Rj**2 * params.l * zp1 / (me * c**2)
    )

    return prefactor_base * nu_array * double_integrals


# ------------------------------------------------------------------ #
#  Single-frequency convenience function                              #
# ------------------------------------------------------------------ #

def synchrotron_luminosity(nu, params, nx: int = 96, ngamma: int = 128):
    """Compute synchrotron νLν at a single frequency. Returns scalar."""
    result = synchrotron_spectrum(jnp.atleast_1d(nu), params, nx, ngamma)
    return result[0]
