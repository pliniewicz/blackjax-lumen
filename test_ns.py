"""
Joint nested sampling: knot + extended jet.
Runs all 4 jet models and compares Bayesian evidence.

Shared (sampled) parameters: G0, p, Lj
Per-component fixed geometry: Rj, l set from observations.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import blackjax
from blackjax.ns.utils import finalise, log_weights, sample, ess
import tqdm

from lumen import (
    make_params, make_cosmology, make_log_likelihood,
    load_sed_Fnu, observed_sed,
    MODEL_1A, MODEL_1B, MODEL_2A, MODEL_2B, MODEL_NAMES,
    kpc,
)


# ------------------------------------------------------------------ #
#  Data                                                               #
# ------------------------------------------------------------------ #

cosmo = make_cosmology(70.0, 0.3, 0.7)
data_knot = load_sed_Fnu("mydata.csv", unit="mJy")
data_ext = load_sed_Fnu("mydata_ext.csv", unit="mJy")

print(f"Knot: {len(data_knot)} points, upper limits: {data_knot.has_upper_limits}")
print(f"Ext:  {len(data_ext)} points, upper limits: {data_ext.has_upper_limits}")

# ------------------------------------------------------------------ #
#  Fixed parameters                                                   #
# ------------------------------------------------------------------ #

FIXED_COMMON = dict(
    q_ratio=1.0, theta=12.0,
    gamma_min=1e2, gamma_max=1e6,
    z=2.5, eta_e=0.1,
)

FIXED_KNOT_GEOM = {"Rj": 10 * kpc, "l": 10 * kpc}
FIXED_EXT_GEOM  = {"Rj": 50 * kpc, "l": 80 * kpc}

# ------------------------------------------------------------------ #
#  Sampled parameters (shared between knot and extended jet)          #
# ------------------------------------------------------------------ #

SAMPLED = {
    "G0":  (2.0,   30.0),
    "p":   (2.0,    3.5),
    "Lj":  (1e46,  1e50),
}

param_names = list(SAMPLED.keys())
ndim = len(param_names)
lo = jnp.array([SAMPLED[k][0] for k in param_names])
hi = jnp.array([SAMPLED[k][1] for k in param_names])


def logprior_fn(x):
    in_bounds = jnp.all((x >= lo) & (x <= hi))
    log_vol = jnp.sum(jnp.log(hi - lo))
    return jnp.where(in_bounds, -log_vol, -jnp.inf)


# ------------------------------------------------------------------ #
#  Run nested sampling for each model                                 #
# ------------------------------------------------------------------ #

n_live = 500
num_delete = 50
num_inner_steps = 5 * ndim

models = [MODEL_1A, MODEL_1B, MODEL_2A, MODEL_2B]
results = {}

for model_id in models:
    model_name = MODEL_NAMES[model_id]
    print(f"\n{'='*60}")
    print(f"  {model_name} (model={model_id})")
    print(f"{'='*60}")

    # Build per-component fixed dicts with this model
    fixed_knot = {**FIXED_COMMON, **FIXED_KNOT_GEOM, "model": int(model_id)}
    fixed_ext  = {**FIXED_COMMON, **FIXED_EXT_GEOM,  "model": int(model_id)}

    def _make_vec_to_params(fixed):
        """Closure to capture the correct fixed dict."""
        def vec_to_params(x):
            kw = dict(fixed)
            for i, name in enumerate(param_names):
                kw[name] = x[i]
            return make_params(**kw)
        return vec_to_params

    vec_to_knot = _make_vec_to_params(fixed_knot)
    vec_to_ext  = _make_vec_to_params(fixed_ext)

    # Build likelihoods (closures capture the model via fixed dicts)
    _loglik_knot = make_log_likelihood(data_knot, cosmo=cosmo, nx=64, ngamma=64)
    _loglik_ext  = make_log_likelihood(data_ext,  cosmo=cosmo, nx=64, ngamma=64)

    def _make_loglik(vk, ve, lk, le):
        def loglikelihood_fn(x):
            return lk(vk(x)) + le(ve(x))
        return loglikelihood_fn

    loglikelihood_fn = _make_loglik(vec_to_knot, vec_to_ext, _loglik_knot, _loglik_ext)

    # Set up sampler
    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
    )

    rng_key = jax.random.PRNGKey(42)
    rng_key, init_key = jax.random.split(rng_key)
    initial_population = jax.random.uniform(
        init_key, (n_live, ndim), minval=lo, maxval=hi,
    )

    state = algo.init(initial_population)
    step = jax.jit(algo.step)
    dead = []

    with tqdm.tqdm(desc=f"{model_name}", unit=" dead") as pbar:
        while not state.integrator.logZ_live - state.integrator.logZ < -3:
            rng_key, subkey = jax.random.split(rng_key)
            state, dead_info = step(subkey, state)
            dead.append(dead_info)
            pbar.update(num_delete)

    rng_key, wkey, skey = jax.random.split(rng_key, 3)
    final = finalise(state, dead)
    ns_ess = ess(skey, final)
    log_w = log_weights(wkey, final, shape=100)
    logzs = jax.scipy.special.logsumexp(log_w, axis=0)
    posterior = sample(skey, final, shape=n_live)

    logZ_mean = float(logzs.mean())
    logZ_std  = float(logzs.std())

    results[model_id] = {
        "name": model_name,
        "logZ": logZ_mean,
        "logZ_err": logZ_std,
        "ess": int(ns_ess),
        "posterior": posterior,
        "final": final,
        "vec_to_knot": vec_to_knot,
        "vec_to_ext": vec_to_ext,
    }

    print(f"  logZ = {logZ_mean:.2f} +/- {logZ_std:.2f}  |  ESS = {int(ns_ess)}")
    for i, name in enumerate(param_names):
        vals = posterior.position[:, i]
        print(f"    {name:10s}: {vals.mean():.4g} +/- {vals.std():.4g}")


# ------------------------------------------------------------------ #
#  Model comparison summary                                           #
# ------------------------------------------------------------------ #

print(f"\n{'='*60}")
print("  MODEL COMPARISON (Bayesian evidence)")
print(f"{'='*60}")

# Sort by logZ (highest = preferred)
ranked = sorted(results.values(), key=lambda r: r["logZ"], reverse=True)
best_logZ = ranked[0]["logZ"]

print(f"  {'Model':<12s} {'logZ':>10s} {'err':>6s} {'Delta logZ':>12s}  {'Interpretation'}")
print(f"  {'-'*12} {'-'*10} {'-'*6} {'-'*12}  {'-'*20}")
for r in ranked:
    delta = r["logZ"] - best_logZ
    if delta == 0:
        interp = "<-- preferred"
    elif delta > -1:
        interp = "indistinguishable"
    elif delta > -2.5:
        interp = "weak evidence against"
    elif delta > -5:
        interp = "moderate evidence against"
    else:
        interp = "strong evidence against"
    print(f"  {r['name']:<12s} {r['logZ']:>10.2f} {r['logZ_err']:>6.2f} {delta:>+12.2f}  {interp}")

# Jeffreys scale reference
print()
print("  Jeffreys scale: |Delta logZ| < 1 inconclusive,")
print("  1-2.5 weak, 2.5-5 moderate, >5 strong")

# ------------------------------------------------------------------ #
#  Corner plots: all models on one figure                             #
# ------------------------------------------------------------------ #

try:
    import anesthetic
    import matplotlib.pyplot as plt

    colors = {"MODEL_1A": "C0", "MODEL_1B": "C1", "MODEL_2A": "C2", "MODEL_2B": "C3"}

    # Overlay all posteriors on the same corner plot
    fig_corner = None
    for r in ranked:
        ns_samples = anesthetic.NestedSamples(
            data=r["final"].particles.position,
            columns=param_names,
            logL=r["final"].particles.loglikelihood,
            logL_birth=r["final"].particles.loglikelihood_birth,
        )
        label = f"{r['name']} (logZ={r['logZ']:.1f})"
        if fig_corner is None:
            fig_corner = ns_samples.plot_2d(
                param_names, label=label, color=colors[r["name"]],
            )
        else:
            ns_samples.plot_2d(
                fig_corner, label=label, color=colors[r["name"]],
            )

    fig_corner.iloc[-1, 0].legend(
        bbox_to_anchor=(len(fig_corner), len(fig_corner)),
        loc="lower right", fontsize=9,
    )
    plt.savefig("ns_corner_all_models.pdf", dpi=150)
    plt.close("all")
    print("\nCorner plot saved to ns_corner_all_models.pdf")
except ImportError:
    print("\nInstall anesthetic for corner plots: uv add anesthetic")

# ------------------------------------------------------------------ #
#  SED plot: best model, both components                              #
# ------------------------------------------------------------------ #

import matplotlib.pyplot as plt

best = ranked[0]
nu_plot = jnp.array(10 ** np.arange(7, 26, 0.05))

median_vec = jnp.median(best["posterior"].position, axis=0)
p_knot = best["vec_to_knot"](median_vec)
p_ext  = best["vec_to_ext"](median_vec)
flux_knot = observed_sed(nu_plot, p_knot, cosmo, nx=64, ngamma=64)
flux_ext  = observed_sed(nu_plot, p_ext,  cosmo, nx=64, ngamma=64)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(nu_plot, flux_knot, "-",  color="C0", lw=1.5, label=f"knot ({best['name']})")
ax.plot(nu_plot, flux_ext,  "--", color="C1", lw=1.5, label=f"extended ({best['name']})")
ax.plot(nu_plot, flux_knot + flux_ext, "-", color="0.4", lw=1, alpha=0.5, label="total")

ax.errorbar(
    np.array(data_knot.frequencies), np.array(data_knot.fluxes),
    yerr=np.array(data_knot.errors),
    fmt="o", color="C0", ms=6, capsize=3, zorder=10, label="knot data",
)

ext_det = data_ext.detections()
ext_ul  = data_ext.upper_limits()
if len(ext_det):
    ax.errorbar(
        np.array(ext_det.frequencies), np.array(ext_det.fluxes),
        yerr=np.array(ext_det.errors),
        fmt="s", color="C1", ms=6, capsize=3, zorder=10, label="ext data",
    )
if len(ext_ul):
    ax.errorbar(
        np.array(ext_ul.frequencies), np.array(ext_ul.fluxes),
        yerr=0.3 * np.array(ext_ul.fluxes),
        fmt="v", color="C1", ms=6, uplims=True, zorder=10, label="ext upper limit",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\nu$ [Hz]", fontsize=13)
ax.set_ylabel(r"$\nu F_\nu$ [erg/s/cm$^2$]", fontsize=13)
ax.set_xlim(1e7, 1e26)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("ns_sed_comparison.pdf", dpi=150)
print(f"\nSED plot (best model: {best['name']}) saved to ns_sed_comparison.pdf")
