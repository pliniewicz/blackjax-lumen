"""Quick test: MAP + a few NS steps for MODEL_1B only."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import blackjax
from blackjax.ns.utils import finalise, log_weights, sample, ess
import tqdm

from lumen import (
    make_params, make_cosmology, make_log_likelihood,
    load_sed_Fnu, MODEL_1B, MODEL_NAMES, kpc,
)

cosmo = make_cosmology(71.0, 0.27, 0.73)
data_knot = load_sed_Fnu("mydata.csv", unit="mJy")
data_ext = load_sed_Fnu("mydata_ext.csv", unit="mJy")

FIXED_COMMON = dict(q_ratio=1.0, p=2.5, gamma_min=10.0, gamma_max=1e5, z=2.5, eta_e=0.1)
FIXED_KNOT_GEOM = {"Rj": 3 * kpc, "l": 3 * kpc}
FIXED_EXT_GEOM  = {"Rj": 3 * kpc, "l": 80 * kpc}

SAMPLED = {
    "theta":         (3.0,  20.0),
    "log10_Lj_knot": (47.0, 49.5),
    "log10_Lj_ext":  (47.0, 49.5),
    "G0_knot":       (3.0,  20.0),
    "G0_ext":        (5.0,  28.0),
}

param_names = list(SAMPLED.keys())
ndim = len(param_names)
lo = jnp.array([SAMPLED[k][0] for k in param_names])
hi = jnp.array([SAMPLED[k][1] for k in param_names])

I_THETA, I_LJK, I_LJE, I_G0K, I_G0E = 0, 1, 2, 3, 4

def _make_vec_to_params(fixed, lj_index, g0_index):
    def vec_to_params(x):
        kw = dict(fixed)
        kw["theta"] = x[I_THETA]
        kw["Lj"]    = 10.0 ** x[lj_index]
        kw["G0"]    = x[g0_index]
        return make_params(**kw)
    return vec_to_params

def logprior_fn(x):
    in_bounds = jnp.all((x >= lo) & (x <= hi))
    log_vol = jnp.sum(jnp.log(hi - lo))
    return jnp.where(in_bounds, -log_vol, -jnp.inf)

def find_map(loglikelihood_fn, x0, bounds_lo, bounds_hi, n_steps=2000, lr=0.01):
    scale = bounds_hi - bounds_lo
    @jax.jit
    def step(x):
        g = jax.grad(loglikelihood_fn)(x)
        g_normalized = g / (jnp.abs(g).max() + 1e-30)
        x_new = x + lr * scale * g_normalized
        margin = 1e-6 * scale
        x_new = jnp.clip(x_new, bounds_lo + margin, bounds_hi - margin)
        return x_new
    x = jnp.array(x0, dtype=jnp.float64)
    best_x = x
    best_ll = loglikelihood_fn(x)
    for i in range(n_steps):
        x = step(x)
        ll = loglikelihood_fn(x)
        improved = ll > best_ll
        best_x = jnp.where(improved, x, best_x)
        best_ll = jnp.where(improved, ll, best_ll)
    return best_x, best_ll

# --- MODEL_1B ---
model_id = MODEL_1B
model_name = MODEL_NAMES[model_id]
fixed_knot = {**FIXED_COMMON, **FIXED_KNOT_GEOM, "model": int(model_id)}
fixed_ext  = {**FIXED_COMMON, **FIXED_EXT_GEOM,  "model": int(model_id)}

vec_to_knot = _make_vec_to_params(fixed_knot, I_LJK, I_G0K)
vec_to_ext  = _make_vec_to_params(fixed_ext,  I_LJE, I_G0E)

_loglik_knot = make_log_likelihood(data_knot, cosmo=cosmo, nx=64, ngamma=64)
_loglik_ext  = make_log_likelihood(data_ext,  cosmo=cosmo, nx=64, ngamma=64)

def _make_loglik(vk, ve, lk, le):
    def loglikelihood_fn(x):
        return lk(vk(x)) + le(ve(x))
    return loglikelihood_fn

loglikelihood_fn = _make_loglik(vec_to_knot, vec_to_ext, _loglik_knot, _loglik_ext)

# MAP
x0 = jnp.array([10.0, 48.3, 48.0, 9.0, 20.0])
map_x, map_ll = find_map(loglikelihood_fn, x0, lo, hi)
print(f"MAP: theta={map_x[0]:.1f}  log10Lj_k={map_x[1]:.2f}  log10Lj_e={map_x[2]:.2f}  "
      f"G0_k={map_x[3]:.1f}  G0_e={map_x[4]:.1f}  ->  logL={map_ll:.2f}")

# Seed 100% near MAP
n_live = 500
num_delete = 50
num_inner_steps = 20 * ndim

rng_key = jax.random.PRNGKey(42)
rng_key, seed_key = jax.random.split(rng_key)
spread = 0.15 * (hi - lo)
initial_population = map_x + spread * jax.random.normal(seed_key, (n_live, ndim))
initial_population = jnp.clip(initial_population, lo, hi)

# Check initial population quality
init_lls = []
for i in range(n_live):
    init_lls.append(float(loglikelihood_fn(initial_population[i])))
init_lls = np.array(init_lls)
print(f"\nInitial population:")
print(f"  median logL = {np.median(init_lls):.1f}")
print(f"  logL > -10: {(init_lls > -10).sum()} ({(init_lls > -10).mean()*100:.0f}%)")
print(f"  logL > -50: {(init_lls > -50).sum()} ({(init_lls > -50).mean()*100:.0f}%)")
print(f"  worst logL: {init_lls.min():.0f}")

# Run NS
algo = blackjax.nss(
    logprior_fn=logprior_fn,
    loglikelihood_fn=loglikelihood_fn,
    num_delete=num_delete,
    num_inner_steps=num_inner_steps,
)

state = algo.init(initial_population)
step_fn = jax.jit(algo.step)
dead = []

print(f"\nRunning NS (n_live={n_live}, num_delete={num_delete}, "
      f"num_inner_steps={num_inner_steps})...")
print(f"Initial: logZ_live={state.integrator.logZ_live:.2f}, logZ={state.integrator.logZ:.2f}")

n_max_steps = 200  # ~10k dead points, should be enough to see if it converges
with tqdm.tqdm(desc=model_name, unit=" dead", total=n_max_steps*num_delete) as pbar:
    for step_i in range(n_max_steps):
        if state.integrator.logZ_live - state.integrator.logZ < -3:
            print(f"\nConverged at step {step_i}!")
            break
        rng_key, subkey = jax.random.split(rng_key)
        state, dead_info = step_fn(subkey, state)
        dead.append(dead_info)
        pbar.update(num_delete)

        if step_i % 20 == 0:
            tqdm.tqdm.write(f"  step {step_i}: logZ_live={state.integrator.logZ_live:.2f}, "
                           f"logZ={state.integrator.logZ:.2f}, "
                           f"diff={state.integrator.logZ_live - state.integrator.logZ:.2f}")

rng_key, wkey, skey = jax.random.split(rng_key, 3)
final = finalise(state, dead)
ns_ess = ess(skey, final)
log_w = log_weights(wkey, final, shape=100)
logzs = jax.scipy.special.logsumexp(log_w, axis=0)
posterior = sample(skey, final, shape=n_live)

logZ_mean = float(logzs.mean())
logZ_std  = float(logzs.std())

print(f"\nlogZ = {logZ_mean:.2f} +/- {logZ_std:.2f}  |  ESS = {int(ns_ess)}")
for i, name in enumerate(param_names):
    vals = posterior.position[:, i]
    print(f"  {name:15s}: {vals.mean():.4g} +/- {vals.std():.4g}")
