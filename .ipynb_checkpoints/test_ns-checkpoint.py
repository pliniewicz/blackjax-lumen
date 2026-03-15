"""
Test nested sampling with lumen using blackjax-ns.

Fits a minimal subset of jet parameters (G0, p, Lj) while holding
the rest fixed. Demonstrates the full pipeline:
    data → log-likelihood → blackjax.nss → posterior + evidence.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import blackjax
from blackjax.ns.utils import finalise, log_weights, sample, ess
import tqdm

from lumen import (
    make_params, make_cosmology, make_log_likelihood,
    load_sed_Fnu, MODEL_1A, kpc,
)


# ------------------------------------------------------------------ #
#  Data & fixed parameters                                            #
# ------------------------------------------------------------------ #

cosmo = make_cosmology(70.0, 0.3, 0.7)
data = load_sed_Fnu("mydata.csv", unit="mJy")

# Fixed (not sampled) parameter values
FIXED = dict(
    q_ratio=1.0, theta=12.0,
    gamma_min=1e2, gamma_max=1e6,
    Rj=10 * kpc, l=10 * kpc,
    z=2.5, eta_e=0.1, model=int(MODEL_1A),
)

# ------------------------------------------------------------------ #
#  Sampled parameters: name, prior bounds (uniform)                   #
# ------------------------------------------------------------------ #

SAMPLED = {
    #  name    (lo,   hi)        — uniform prior in these ranges
    "G0":     (2.0,  30.0),
    "p":      (2.0,   3.5),
    "Lj":     (1e46, 1e50),
}

param_names = list(SAMPLED.keys())
ndim = len(param_names)
lo = jnp.array([SAMPLED[k][0] for k in param_names])
hi = jnp.array([SAMPLED[k][1] for k in param_names])

# ------------------------------------------------------------------ #
#  Map flat vector ↔ JetParams                                        #
# ------------------------------------------------------------------ #

def vec_to_params(x):
    """Map a flat array of sampled values → JetParams pytree."""
    kw = dict(FIXED)
    for i, name in enumerate(param_names):
        kw[name] = x[i]
    return make_params(**kw)


# ------------------------------------------------------------------ #
#  Log-prior (uniform box)                                            #
# ------------------------------------------------------------------ #

def logprior_fn(x):
    """Uniform prior: 0 inside the box, -inf outside."""
    in_bounds = jnp.all((x >= lo) & (x <= hi))
    log_vol = jnp.sum(jnp.log(hi - lo))
    return jnp.where(in_bounds, -log_vol, -jnp.inf)


# ------------------------------------------------------------------ #
#  Log-likelihood  (wraps lumen's SED model)                          #
# ------------------------------------------------------------------ #

# Build the compiled likelihood closure once
_loglik = make_log_likelihood(data, cosmo=cosmo, nx=64, ngamma=64)

def loglikelihood_fn(x):
    params = vec_to_params(x)
    return _loglik(params)


# ------------------------------------------------------------------ #
#  Nested sampling                                                    #
# ------------------------------------------------------------------ #

n_live = 500
num_delete = 50
num_inner_steps = 5 * ndim

algo = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps,
)

# Initialise live points from the prior (uniform box)
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)
initial_population = jax.random.uniform(
    init_key, (n_live, ndim), minval=lo, maxval=hi,
)

state = algo.init(initial_population)
step = jax.jit(algo.step)

dead = []

print(f"Running nested sampling: {ndim} dims, {n_live} live points")
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not state.integrator.logZ_live - state.integrator.logZ < -3:
        rng_key, subkey = jax.random.split(rng_key)
        state, dead_info = step(subkey, state)
        dead.append(dead_info)
        pbar.update(num_delete)

# ------------------------------------------------------------------ #
#  Results                                                            #
# ------------------------------------------------------------------ #

rng_key, wkey, skey = jax.random.split(rng_key, 3)
final = finalise(state, dead)
ns_ess = ess(skey, final)
log_w = log_weights(wkey, final, shape=100)
logzs = jax.scipy.special.logsumexp(log_w, axis=0)
posterior = sample(skey, final, shape=n_live)

print(f"\nlogZ = {logzs.mean():.2f} ± {logzs.std():.2f}")
print(f"ESS  = {int(ns_ess)}")
print()
for i, name in enumerate(param_names):
    vals = posterior.position[:, i]
    print(f"  {name:10s}: {vals.mean():.4g} ± {vals.std():.4g}")

# ------------------------------------------------------------------ #
#  Corner plot (optional, requires anesthetic)                        #
# ------------------------------------------------------------------ #

try:
    import anesthetic
    import matplotlib.pyplot as plt

    ns_samples = anesthetic.NestedSamples(
        data=final.particles.position,
        columns=param_names,
        logL=final.particles.loglikelihood,
        logL_birth=final.particles.loglikelihood_birth,
    )

    prior_plot = ns_samples.set_beta(0.0).plot_2d(
        param_names, label="prior"
    )
    ns_samples.plot_2d(prior_plot, label="posterior")
    prior_plot.iloc[-1, 0].legend(
        bbox_to_anchor=(len(prior_plot), len(prior_plot)),
        loc="lower right",
    )
    plt.savefig("ns_corner.pdf", dpi=150)
    print("\nCorner plot saved to ns_corner.pdf")
except ImportError:
    print("\nInstall anesthetic for corner plots: uv add anesthetic")
