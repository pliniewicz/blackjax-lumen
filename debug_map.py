"""Debug the MAP finder: why does gradient ascent fail?"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from lumen import (
    make_params, make_cosmology, make_log_likelihood,
    load_sed_Fnu, MODEL_1B, kpc,
)

cosmo = make_cosmology(71.0, 0.27, 0.73)
data_knot = load_sed_Fnu("mydata.csv", unit="mJy")
data_ext = load_sed_Fnu("mydata_ext.csv", unit="mJy")

FIXED_COMMON = dict(q_ratio=1.0, p=2.5, gamma_min=10.0, gamma_max=1e5, z=2.5, eta_e=0.1)

# Use user's Rj=3 kpc for knot
fixed_knot = {**FIXED_COMMON, "Rj": 3*kpc, "l": 3*kpc, "model": int(MODEL_1B)}
fixed_ext  = {**FIXED_COMMON, "Rj": 3*kpc, "l": 80*kpc, "model": int(MODEL_1B)}

loglik_knot = make_log_likelihood(data_knot, cosmo=cosmo, nx=64, ngamma=64)
loglik_ext  = make_log_likelihood(data_ext,  cosmo=cosmo, nx=64, ngamma=64)

I_THETA, I_LJ, I_G0K, I_G0E = 0, 1, 2, 3

def _make_vec_to_params(fixed, g0_index):
    def vec_to_params(x):
        kw = dict(fixed)
        kw["theta"] = x[I_THETA]
        kw["Lj"]    = 10.0 ** x[I_LJ]
        kw["G0"]    = x[g0_index]
        return make_params(**kw)
    return vec_to_params

vec_to_knot = _make_vec_to_params(fixed_knot, I_G0K)
vec_to_ext  = _make_vec_to_params(fixed_ext,  I_G0E)

def loglikelihood_fn(x):
    return loglik_knot(vec_to_knot(x)) + loglik_ext(vec_to_ext(x))

# Test: evaluate at known good point
x_good = jnp.array([10.0, 48.0, 9.0, 18.0])
ll_good = float(loglikelihood_fn(x_good))
print(f"logL at x_good = {ll_good:.4f}")

# Test: evaluate gradient at known good point
grad_fn = jax.grad(loglikelihood_fn)
g = grad_fn(x_good)
print(f"grad at x_good = {g}")
print(f"any NaN in grad? {jnp.any(jnp.isnan(g))}")

# Now run MAP step by step
lo = jnp.array([2.0, 44.0, 2.0, 2.0])
hi = jnp.array([34.0, 50.0, 30.0, 30.0])

x0 = jnp.array([7.5, 47.5, 9.0, 15.0])
print(f"\nlogL at x0 = {float(loglikelihood_fn(x0)):.4f}")
g0 = grad_fn(x0)
print(f"grad at x0 = {g0}")
print(f"any NaN? {jnp.any(jnp.isnan(g0))}")

# Manual gradient ascent with verbose output
lr = 0.05
x = x0.copy()
print(f"\n=== Manual gradient ascent (lr={lr}) ===")
for i in range(20):
    ll = float(loglikelihood_fn(x))
    g = grad_fn(x)
    has_nan = bool(jnp.any(jnp.isnan(g)))
    has_inf = bool(jnp.any(jnp.isinf(g)))
    g_norm = float(jnp.linalg.norm(g))

    print(f"  step {i:3d}: x=[{x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}, {x[3]:.3f}]  "
          f"logL={ll:.2f}  |grad|={g_norm:.1f}  nan={has_nan}  inf={has_inf}")

    if has_nan or has_inf:
        print(f"    grad = {g}")
        break

    x_new = x + lr * g
    margin = 1e-6 * (hi - lo)
    x = jnp.clip(x_new, lo + margin, hi - margin)

# Try with smaller lr
print(f"\n=== With lr=0.001 ===")
x = x0.copy()
for i in range(20):
    ll = float(loglikelihood_fn(x))
    g = grad_fn(x)
    g_norm = float(jnp.linalg.norm(g))
    has_nan = bool(jnp.any(jnp.isnan(g)))

    print(f"  step {i:3d}: x=[{x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}, {x[3]:.3f}]  "
          f"logL={ll:.2f}  |grad|={g_norm:.1f}  nan={has_nan}")

    if has_nan:
        print(f"    grad = {g}")
        break

    x_new = x + 0.001 * g
    margin = 1e-6 * (hi - lo)
    x = jnp.clip(x_new, lo + margin, hi - margin)
