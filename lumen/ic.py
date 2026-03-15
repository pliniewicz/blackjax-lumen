"""
Inverse Compton scattering on CMB photons — fully vectorized JAX.

Mirrors ic.jl (Jones approximation with optional Klein-Nishina corrections).
Same tensorized (x, γ) grid strategy as synchrotron.py, vmapped over ν.

Reference:
    Jones (1968), Blumenthal & Gould (1970)
"""

import jax
import jax.numpy as jnp
from functools import partial

from .constants import me, c, h, sigma_T, U_CMB, eps_CMB
from .integration import get_gl, map_to_interval, gl_quad
from .profiles import Gamma_profile, Doppler_factor, PB1
from .electrons import electron_distribution, electron_norm, K_e


# ------------------------------------------------------------------ #
#  IC kernel                                                          #
# ------------------------------------------------------------------ #

def ic_kernel(nu, eps0, gamma, use_kn: bool = True):
    """
    Inverse Compton kernel (Jones approximation).

    Parameters
    ----------
    nu : scattered photon frequency [Hz] (in the relevant frame)
    eps0 : seed photon energy [dimensionless, in me c² units]
    gamma : electron Lorentz factor
    use_kn : include Klein-Nishina corrections

    Returns
    -------
    Kernel value (scalar).

    All branching uses jnp.where for JAX traceability.
    """
    eps = h * nu / (me * c ** 2)  # scattered photon energy, dimensionless

    Gamma_e = 4.0 * eps0 * gamma  # energy transfer parameter

    # Compton kinematics
    denom = 4.0 * eps0 * gamma * (gamma - eps)
    # Guard division by zero
    denom_safe = jnp.where(jnp.abs(denom) < 1e-300, 1e-300, denom)
    q_ic = eps / denom_safe

    # Kinematic limits: q_ic must be in (1/(4γ²), 1)
    q_min = 1.0 / (4.0 * gamma ** 2)
    in_bounds = (q_ic > q_min) & (q_ic < 1.0)

    # Thomson regime
    thomson = 2.0 * q_ic * jnp.log(jnp.maximum(q_ic, 1e-300)) + q_ic + 1.0 - 2.0 * q_ic ** 2

    # Klein-Nishina correction
    kn_num = (Gamma_e * q_ic) ** 2 * (1.0 - q_ic)
    kn_den = 2.0 * (1.0 + Gamma_e * q_ic)
    kn = jnp.where(use_kn, kn_num / kn_den, 0.0)

    result = thomson + kn
    return jnp.where(in_bounds, result, 0.0)


# ------------------------------------------------------------------ #
#  Precomputation (extends synchrotron pattern with seed photon energy)
# ------------------------------------------------------------------ #

def _precompute_x_quantities_ic(x_nodes, params, pb1, enorm):
    """
    x-dependent quantities for IC: δ(x), K_e(x), Γ(x), ε0_jet(x).
    """
    model = int(params.model)
    zp1 = params.z + 1.0

    def _single_x(x):
        Gx = Gamma_profile(x, params.G0)
        dx = Doppler_factor(x, params.G0, params.theta)
        Kx = K_e(x, params, pb1, enorm)
        eps0_jet = zp1 * Gx * eps_CMB  # seed photon energy boosted to jet frame
        return dx, Kx, eps0_jet

    delta_xs, K_xs, eps0s = jax.vmap(_single_x)(x_nodes)
    return delta_xs, K_xs, eps0s


# ------------------------------------------------------------------ #
#  Core spectrum                                                      #
# ------------------------------------------------------------------ #

@partial(jax.jit, static_argnums=(2, 3, 4))
def ic_spectrum(nu_array, params, nx: int = 96, ngamma: int = 128,
                use_kn: bool = True):
    """
    Compute IC/CMB νLν [erg/s] at each frequency in nu_array.

    Parameters
    ----------
    nu_array : jax array, shape (N,)
        Observer-frame frequencies [Hz].
    params : JetParams
    nx, ngamma : int
        Quadrature orders.
    use_kn : bool
        Include Klein-Nishina corrections.

    Returns
    -------
    nuLnu : jax array, shape (N,)
    """
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

    # Precompute
    delta_xs, K_xs, eps0s = _precompute_x_quantities_ic(
        x_nodes, params, pb1, enorm
    )
    gammas = jnp.exp(log_gamma_nodes)
    f_es = electron_distribution(gammas, params.p, params.gamma_min, params.gamma_max)

    # Prefactor: 3π h² σ_T Rj² l U_CMB (1+z)⁴ / (2 me² c³ ε_CMB²)
    prefactor_base = (
        3.0 * jnp.pi * h ** 2 * sigma_T
        * params.Rj ** 2 * params.l * U_CMB * zp1 ** 4
        / (2.0 * me ** 2 * c ** 3 * eps_CMB ** 2)
    )

    def _integrand_single_nu(nu):
        # IC kernel on (nx, ng) grid
        # nu in jet comoving frame: zp1 * nu / δ(x)
        nu_jet = zp1 * nu / delta_xs[:, None]  # (nx, 1)
        eps0_grid = eps0s[:, None]               # (nx, 1)
        gamma_grid = gammas[None, :]             # (1, ng)

        f_ic = ic_kernel(nu_jet, eps0_grid, gamma_grid, use_kn)  # (nx, ng)

        # Integrand: x · δ(x) · K_e(x) · f_e(γ) · f_ic / γ²  · γ [Jacobian]
        # The γ from Jacobian and 1/γ² give 1/γ net
        integrand = (
            x_nodes[:, None]
            * delta_xs[:, None]
            * K_xs[:, None]
            * f_es[None, :]
            * f_ic
            / gammas[None, :]   # net 1/γ = γ_jacobian / γ²
        )

        inner = jnp.dot(integrand, weights_g) * sg   # (nx,)
        outer = jnp.dot(weights_x, inner) * sx        # scalar
        return outer

    double_integrals = jax.vmap(_integrand_single_nu)(nu_array)

    return prefactor_base * nu_array ** 2 * double_integrals


def ic_luminosity(nu, params, nx: int = 96, ngamma: int = 128,
                  use_kn: bool = True):
    """IC νLν at a single frequency. Returns scalar."""
    result = ic_spectrum(jnp.atleast_1d(nu), params, nx, ngamma, use_kn)
    return result[0]
