"""
Gauss-Legendre quadrature — JAX compatible.

Nodes/weights are precomputed once via NumPy (exact). The quadrature
sums are pure JAX operations so they participate in autodiff.

Design note: we intentionally keep this module simple. The actual
double-integral structure (x × γ grid) is handled in the emission
modules, where we can exploit the separable structure and vmap
efficiently.
"""

from numpy.polynomial.legendre import leggauss
import numpy as np
import jax.numpy as jnp
from functools import lru_cache


@lru_cache(maxsize=32)
def _gl_numpy(n: int):
    """Cache GL nodes/weights as plain NumPy (never traced)."""
    nodes, weights = leggauss(n)
    return nodes, weights


def get_gl(n: int = 128):
    """GL nodes and weights as fresh JAX arrays (safe inside jit)."""
    nodes, weights = _gl_numpy(n)
    return jnp.array(nodes, dtype=jnp.float64), jnp.array(weights, dtype=jnp.float64)


# Default order
GL_N = 128


def gl_quad(f_vals, weights, scale):
    """
    Contract pre-evaluated integrand values with GL weights.

    Parameters
    ----------
    f_vals : array, shape (n,) or (n, ...)
        Integrand evaluated at the GL nodes.
    weights : array, shape (n,)
        GL weights.
    scale : float
        Half-width of the integration interval, (b-a)/2.

    Returns
    -------
    Scalar (or array if f_vals has trailing dims).
    """
    return scale * jnp.dot(weights, f_vals)


def map_to_interval(nodes, a, b):
    """Map GL nodes from [-1, 1] to [a, b]. Returns (mapped_nodes, scale)."""
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    return scale * nodes + shift, scale
