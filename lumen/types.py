"""
Jet parameter containers — JAX pytree-compatible.

Model indices:
    0 = MODEL_1A    1 = MODEL_1B
    2 = MODEL_2A    3 = MODEL_2B

'A' models (0, 2) use magnetic pressure for K_e normalization.
'B' models (1, 3) use total pressure.

Design note:
    `model` is a *discrete* label, not a continuous parameter.
    We register a custom pytree so that `model` lives in the
    auxiliary (static) dict — JAX recompiles when model changes,
    but never tries to trace through it.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass

# Model enum as plain ints
MODEL_1A, MODEL_1B = 0, 1
MODEL_2A, MODEL_2B = 2, 3

MODEL_NAMES = {
    0: "MODEL_1A", 1: "MODEL_1B",
    2: "MODEL_2A", 3: "MODEL_2B",
}

# Fields that JAX should trace (differentiable)
_TRACED_FIELDS = (
    "G0", "q_ratio", "p", "theta",
    "gamma_min", "gamma_max",
    "Rj", "Lj", "l", "z", "eta_e",
)

# Fields that are static (trigger recompilation, not traced)
_STATIC_FIELDS = ("model",)


@dataclass
class JetParams:
    """
    All physical parameters for one jet component.

    Registered as a JAX pytree with `model` as static auxiliary data.
    All other fields are traced leaves (support grad, vmap, etc.).
    """
    G0: float
    q_ratio: float
    p: float
    theta: float
    gamma_min: float
    gamma_max: float
    Rj: float
    Lj: float
    l: float
    z: float
    eta_e: float
    model: int = 1  # MODEL_1B default, STATIC


def _jetparams_flatten(p):
    children = tuple(getattr(p, f) for f in _TRACED_FIELDS)
    aux = {f: getattr(p, f) for f in _STATIC_FIELDS}
    return children, aux


def _jetparams_unflatten(aux, children):
    kwargs = dict(zip(_TRACED_FIELDS, children))
    kwargs.update(aux)
    return JetParams(**kwargs)


jax.tree_util.register_pytree_node(
    JetParams,
    _jetparams_flatten,
    _jetparams_unflatten,
)


def make_params(G0, q_ratio, p, theta, gamma_min, gamma_max,
                Rj, Lj, l, z, eta_e, model) -> JetParams:
    """Construct JetParams with explicit float64 promotion."""
    return JetParams(
        G0=jnp.float64(G0),
        q_ratio=jnp.float64(q_ratio),
        p=jnp.float64(p),
        theta=jnp.float64(theta),
        gamma_min=jnp.float64(gamma_min),
        gamma_max=jnp.float64(gamma_max),
        Rj=jnp.float64(Rj),
        Lj=jnp.float64(Lj),
        l=jnp.float64(l),
        z=jnp.float64(z),
        eta_e=jnp.float64(eta_e),
        model=int(model),
    )
