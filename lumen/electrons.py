"""
Electron energy distributions and normalization.

Mirrors electrons.jl. All functions are JAX-traceable.
"""

import jax
import jax.numpy as jnp

from .constants import me, c
from .profiles import f_profile, p_profile, PB1
from .integration import get_gl, map_to_interval, gl_quad


# ------------------------------------------------------------------ #
#  Distributions                                                      #
# ------------------------------------------------------------------ #

def simple_electron_distribution(gamma, p_index):
    """Simple power-law: γ^{-p}."""
    return gamma ** (-p_index)


def electron_distribution(gamma, p_index, gamma_min, gamma_max):
    """
    Smoothed power-law with exponential cutoffs:
        f_e(γ) = γ^{-p} · exp(-γ/γ_max - γ_min/γ)
    """
    return gamma ** (-p_index)# * jnp.exp(-gamma / gamma_max - gamma_min / gamma)


# ------------------------------------------------------------------ #
#  Normalization                                                      #
# ------------------------------------------------------------------ #

def electron_norm(params, n: int = 128):
    """
    ∫ γ · f_e(γ) dγ  over [γ_min, γ_max].

    Integration limits match the physical support of the pure power-law
    distribution exactly.  If exponential cutoffs are re-enabled in
    electron_distribution, extend to [γ_min/2, 3·γ_max] accordingly.

    Depends only on spectral parameters (p, γ_min, γ_max), not on
    position x.  Precompute once per parameter set.
    """
    log_a = jnp.log(params.gamma_min)
    log_b = jnp.log(params.gamma_max)

    nodes, weights = get_gl(n)
    log_gammas, scale = map_to_interval(nodes, log_a, log_b)
    gammas = jnp.exp(log_gammas)

    # Integrand: γ · f_e(γ) · (dγ = γ d(log γ))  →  γ² · f_e(γ)
    f_e = electron_distribution(gammas, params.p, params.gamma_min, params.gamma_max)
    f_vals = gammas * gammas * f_e      # γ · f_e · γ  (Jacobian from log transform)

    return gl_quad(f_vals, weights, scale)


def K_e(x, params, pb1, enorm):
    """
    Electron normalization at radial position x.

    A-models (0, 2):  K_e ∝ η_e · f(x) · PB1
    B-models (1, 3):  K_e ∝ η_e · p(x) · PB1 / q
    """
    mec2 = me * c ** 2
    model = int(params.model)  # ensure Python int, not tracer

    # if model in (0, 2):
    if model in (0, 2):
        # A-models: magnetic pressure normalization
        fp = f_profile(x, params.G0, model)
        return params.eta_e * fp * pb1 / (mec2 * enorm)
    else:
        # B-models: total pressure normalization
        pp = p_profile(x, params.G0, params.q_ratio, model)
        return params.eta_e * pp * pb1 / (params.q_ratio * mec2 * enorm)
