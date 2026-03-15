"""Test the fixed MAP finder with normalized gradients."""
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
fixed_knot = {**FIXED_COMMON, "Rj": 3*kpc, "l": 3*kpc, "model": int(MODEL_1B)}
fixed_ext  = {**FIXED_COMMON, "Rj": 3*kpc, "l": 80*kpc, "model": int(MODEL_1B)}

loglik_knot = make_log_likelihood(data_knot, cosmo=cosmo, nx=64, ngamma=64)
loglik_ext  = make_log_likelihood(data_ext,  cosmo=cosmo, nx=64, ngamma=64)

I_THETA, I_LJK, I_LJE, I_G0K, I_G0E = 0, 1, 2, 3, 4

def _make_vec_to_params(fixed, lj_index, g0_index):
    def vec_to_params(x):
        kw = dict(fixed)
        kw["theta"] = x[I_THETA]
        kw["Lj"]    = 10.0 ** x[lj_index]
        kw["G0"]    = x[g0_index]
        return make_params(**kw)
    return vec_to_params

vec_to_knot = _make_vec_to_params(fixed_knot, I_LJK, I_G0K)
vec_to_ext  = _make_vec_to_params(fixed_ext,  I_LJE, I_G0E)

def loglikelihood_fn(x):
    return loglik_knot(vec_to_knot(x)) + loglik_ext(vec_to_ext(x))

lo = jnp.array([2.0, 46.0, 46.0, 2.0, 2.0])
hi = jnp.array([30.0, 50.0, 50.0, 30.0, 30.0])
scale = hi - lo

# User's best: theta=10, Lj_knot=2e48, Lj_ext=1e48, G0_knot=9, G0_ext=20
x_user = jnp.array([10.0, np.log10(2e48), np.log10(1e48), 9.0, 20.0])
print(f"User's best: x={x_user}")
print(f"  logL = {float(loglikelihood_fn(x_user)):.4f}")

# MAP with normalized gradient
x0 = jnp.array([10.0, 48.3, 48.0, 9.0, 20.0])
lr = 0.01

print(f"\nStarting MAP from x0={x0}")
print(f"  logL(x0) = {float(loglikelihood_fn(x0)):.4f}")

grad_fn = jax.grad(loglikelihood_fn)
x = x0.copy()
best_x = x
best_ll = loglikelihood_fn(x)

print(f"\n{'step':>5s}  {'theta':>6s} {'lLj_k':>7s} {'lLj_e':>7s} {'G0_k':>6s} {'G0_e':>6s}  {'logL':>10s}")
for i in range(2000):
    g = grad_fn(x)
    g_norm = g / (jnp.abs(g).max() + 1e-30)
    x_new = x + lr * scale * g_norm
    margin = 1e-6 * scale
    x = jnp.clip(x_new, lo + margin, hi - margin)

    ll = loglikelihood_fn(x)
    improved = ll > best_ll
    best_x = jnp.where(improved, x, best_x)
    best_ll = jnp.where(improved, ll, best_ll)

    if i < 20 or i % 100 == 0 or i == 1999:
        print(f"{i:5d}  {x[0]:6.2f} {x[1]:7.3f} {x[2]:7.3f} {x[3]:6.2f} {x[4]:6.2f}  {float(ll):10.4f}")

print(f"\nMAP result:")
print(f"  x = [{best_x[0]:.2f}, {best_x[1]:.3f}, {best_x[2]:.3f}, {best_x[3]:.2f}, {best_x[4]:.2f}]")
print(f"  logL = {float(best_ll):.4f}")

# Compare SED at MAP vs user's best
print(f"\n=== SED at MAP ===")
pk = vec_to_knot(best_x)
pe = vec_to_ext(best_x)
from lumen import observed_sed
for pt in data_knot.points:
    nu = jnp.array([pt.nu])
    flux = float(observed_sed(nu, pk, cosmo, 64, 64)[0])
    print(f"  Knot: nu={pt.nu:.2e}  model={flux:.3e}  data={pt.nuFnu:.3e}  ratio={flux/pt.nuFnu:.3f}")
for pt in data_ext.points:
    nu = jnp.array([pt.nu])
    flux = float(observed_sed(nu, pe, cosmo, 64, 64)[0])
    ul = " [UL]" if pt.is_upper else ""
    print(f"  Ext:  nu={pt.nu:.2e}  model={flux:.3e}  data={pt.nuFnu:.3e}  ratio={flux/pt.nuFnu:.3f}{ul}")
