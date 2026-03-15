"""
Debug: compare logL computed via grid scan (make_params directly)
vs the test_ns.py pipeline (vec_to_params -> loglik).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from lumen import (
    make_params, make_cosmology, make_log_likelihood,
    load_sed_Fnu, observed_sed,
    MODEL_1B, kpc,
)

cosmo = make_cosmology(71.0, 0.27, 0.73)
data_knot = load_sed_Fnu("mydata.csv", unit="mJy")
data_ext = load_sed_Fnu("mydata_ext.csv", unit="mJy")

# ---- Method 1: Direct (like grid scan) ----
FIXED_COMMON = dict(q_ratio=1.0, p=2.5, gamma_min=10.0, gamma_max=1e5, z=2.5, eta_e=0.1)
FIXED_KNOT = {**FIXED_COMMON, "Rj": 1.5*kpc, "l": 3*kpc, "model": int(MODEL_1B)}
FIXED_EXT  = {**FIXED_COMMON, "Rj": 3*kpc,   "l": 80*kpc, "model": int(MODEL_1B)}

loglik_knot = make_log_likelihood(data_knot, cosmo=cosmo, nx=64, ngamma=64)
loglik_ext  = make_log_likelihood(data_ext,  cosmo=cosmo, nx=64, ngamma=64)

# Test point: theta=10, log10_Lj=48.0, G0_knot=9, G0_ext=18
theta, log10_Lj, g0k, g0e = 10.0, 48.0, 9.0, 18.0

pk_direct = make_params(**FIXED_KNOT, theta=theta, Lj=10.0**log10_Lj, G0=g0k)
pe_direct = make_params(**FIXED_EXT,  theta=theta, Lj=10.0**log10_Lj, G0=g0e)

ll_knot_direct = float(loglik_knot(pk_direct))
ll_ext_direct  = float(loglik_ext(pe_direct))
print(f"=== Method 1: Direct make_params ===")
print(f"  logL_knot = {ll_knot_direct:.4f}")
print(f"  logL_ext  = {ll_ext_direct:.4f}")
print(f"  logL_joint = {ll_knot_direct + ll_ext_direct:.4f}")

# ---- Method 2: Via vec_to_params (like test_ns.py) ----
I_THETA, I_LJ, I_G0K, I_G0E = 0, 1, 2, 3

def _make_vec_to_params(fixed, g0_index):
    def vec_to_params(x):
        kw = dict(fixed)
        kw["theta"] = x[I_THETA]
        kw["Lj"]    = 10.0 ** x[I_LJ]
        kw["G0"]    = x[g0_index]
        return make_params(**kw)
    return vec_to_params

# NOTE: test_ns.py builds fixed dicts as:
#   fixed_knot = {**FIXED_COMMON, **FIXED_KNOT_GEOM, "model": int(model_id)}
# where FIXED_KNOT_GEOM = {"Rj": 1.5 * kpc, "l": 3 * kpc}
# So fixed_knot does NOT contain "model" from FIXED_KNOT above, it's added separately

fixed_knot_testns = {**FIXED_COMMON, "Rj": 1.5*kpc, "l": 3*kpc, "model": int(MODEL_1B)}
fixed_ext_testns  = {**FIXED_COMMON, "Rj": 3*kpc,   "l": 80*kpc, "model": int(MODEL_1B)}

vec_to_knot = _make_vec_to_params(fixed_knot_testns, I_G0K)
vec_to_ext  = _make_vec_to_params(fixed_ext_testns,  I_G0E)

x = jnp.array([theta, log10_Lj, g0k, g0e])

pk_vec = vec_to_knot(x)
pe_vec = vec_to_ext(x)

print(f"\n=== Method 2: vec_to_params ===")
print(f"  x = {x}")

# Compare the params objects
print(f"\n  Knot params comparison:")
print(f"    Direct: G0={pk_direct.G0}, theta={pk_direct.theta}, Lj={pk_direct.Lj:.3e}, "
      f"Rj={pk_direct.Rj:.3e}, l={pk_direct.l:.3e}, model={pk_direct.model}")
print(f"    Vec:    G0={pk_vec.G0}, theta={pk_vec.theta}, Lj={pk_vec.Lj:.3e}, "
      f"Rj={pk_vec.Rj:.3e}, l={pk_vec.l:.3e}, model={pk_vec.model}")

print(f"\n  Ext params comparison:")
print(f"    Direct: G0={pe_direct.G0}, theta={pe_direct.theta}, Lj={pe_direct.Lj:.3e}, "
      f"Rj={pe_direct.Rj:.3e}, l={pe_direct.l:.3e}, model={pe_direct.model}")
print(f"    Vec:    G0={pe_vec.G0}, theta={pe_vec.theta}, Lj={pe_vec.Lj:.3e}, "
      f"Rj={pe_vec.Rj:.3e}, l={pe_vec.l:.3e}, model={pe_vec.model}")

ll_knot_vec = float(loglik_knot(pk_vec))
ll_ext_vec  = float(loglik_ext(pe_vec))
print(f"\n  logL_knot = {ll_knot_vec:.4f}")
print(f"  logL_ext  = {ll_ext_vec:.4f}")
print(f"  logL_joint = {ll_knot_vec + ll_ext_vec:.4f}")

# ---- Method 3: The full joint loglik function (as used in test_ns.py) ----
def _make_loglik(vk, ve, lk, le):
    def loglikelihood_fn(x):
        return lk(vk(x)) + le(ve(x))
    return loglikelihood_fn

loglikelihood_fn = _make_loglik(vec_to_knot, vec_to_ext, loglik_knot, loglik_ext)
ll_joint = float(loglikelihood_fn(x))
print(f"\n=== Method 3: Joint loglik function ===")
print(f"  logL_joint = {ll_joint:.4f}")

# ---- Check SED values at this point ----
print(f"\n=== SED comparison ===")
for label, params in [("Knot (direct)", pk_direct), ("Knot (vec)", pk_vec),
                       ("Ext (direct)", pe_direct), ("Ext (vec)", pe_vec)]:
    data = data_knot if "Knot" in label else data_ext
    for pt in data.points:
        nu = jnp.array([pt.nu])
        flux = float(observed_sed(nu, params, cosmo, 64, 64)[0])
        print(f"  {label}: nu={pt.nu:.2e} -> nuFnu={flux:.3e} (data={pt.nuFnu:.3e})")

# ---- Now test the user's hand-tuned params ----
print(f"\n=== User's hand-tuned params (MODEL_1B) ===")
p_knot_user = make_params(
    G0=9.0, q_ratio=1.0, p=2.5, theta=10.0,
    gamma_min=1e1, gamma_max=1e5,
    Rj=3*kpc, Lj=2e48, l=3*kpc,
    z=2.5, eta_e=0.1, model=int(MODEL_1B),
)
p_ext_user = make_params(
    G0=20.0, q_ratio=1.0, p=2.5, theta=10.0,
    gamma_min=1e1, gamma_max=1e5,
    Rj=3*kpc, Lj=1e48, l=80*kpc,
    z=2.5, eta_e=0.1, model=int(MODEL_1B),
)

ll_knot_user = float(loglik_knot(p_knot_user))
ll_ext_user  = float(loglik_ext(p_ext_user))
print(f"  logL_knot = {ll_knot_user:.4f}")
print(f"  logL_ext  = {ll_ext_user:.4f}")
print(f"  logL_joint = {ll_knot_user + ll_ext_user:.4f}")

for pt in data_knot.points:
    nu = jnp.array([pt.nu])
    flux = float(observed_sed(nu, p_knot_user, cosmo, 64, 64)[0])
    print(f"  Knot: nu={pt.nu:.2e} -> model={flux:.3e} data={pt.nuFnu:.3e} ratio={flux/pt.nuFnu:.3f}")

for pt in data_ext.points:
    nu = jnp.array([pt.nu])
    flux = float(observed_sed(nu, p_ext_user, cosmo, 64, 64)[0])
    ul = " [UL]" if pt.is_upper else ""
    print(f"  Ext:  nu={pt.nu:.2e} -> model={flux:.3e} data={pt.nuFnu:.3e} ratio={flux/pt.nuFnu:.3f}{ul}")

# ---- CRITICAL: Does the user use DIFFERENT Lj for knot vs ext? ----
# The user has Lj=2e48 for knot and Lj=1e48 for ext
# But in test_ns.py, log10_Lj is SHARED! log10(2e48)=48.30, log10(1e48)=48.0
# This means the joint fit with shared Lj CANNOT match the user's best params!
print(f"\n=== CRITICAL: User uses different Lj for knot ({2e48:.1e}) vs ext ({1e48:.1e}) ===")
print(f"  log10(Lj_knot) = {np.log10(2e48):.2f}")
print(f"  log10(Lj_ext)  = {np.log10(1e48):.2f}")
print(f"  These differ by 0.30 dex — shared log10_Lj CANNOT match both!")

# Also note: user uses Rj=3 kpc for knot, not 1.5 kpc!
print(f"\n=== User uses Rj=3 kpc for knot, test_ns uses Rj=1.5 kpc ===")
