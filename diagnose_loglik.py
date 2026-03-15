"""
Quick diagnostic: evaluate knot and ext likelihoods separately
to understand where the joint optimum is.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from lumen import (
    make_params, make_cosmology, make_log_likelihood,
    load_sed_Fnu, observed_sed,
    MODEL_1A, MODEL_NAMES, kpc,
)

cosmo = make_cosmology(70.0, 0.3, 0.7)
data_knot = load_sed_Fnu("mydata.csv", unit="mJy")
data_ext = load_sed_Fnu("mydata_ext.csv", unit="mJy")

print("=== Data summary ===")
print(f"Knot frequencies: {np.array(data_knot.frequencies)}")
print(f"Knot log10(nuFnu): {np.array(data_knot.log_fluxes)}")
print(f"Knot log10 errors: {np.array(data_knot.log_errors)}")
print(f"Ext  frequencies: {np.array(data_ext.frequencies)}")
print(f"Ext  log10(nuFnu): {np.array(data_ext.log_fluxes)}")
print(f"Ext  log10 errors: {np.array(data_ext.log_errors)}")
print(f"Ext  upper limits: {np.array(data_ext.upper_limit_mask)}")

# Fixed params for MODEL_1A
FIXED_COMMON = dict(q_ratio=1.0, p=2.5, gamma_min=1e2, gamma_max=1e6, z=2.5, eta_e=0.1)
FIXED_KNOT = {**FIXED_COMMON, "Rj": 10*kpc, "l": 10*kpc, "model": int(MODEL_1A)}
FIXED_EXT  = {**FIXED_COMMON, "Rj": 50*kpc, "l": 80*kpc, "model": int(MODEL_1A)}

loglik_knot = make_log_likelihood(data_knot, cosmo=cosmo, nx=64, ngamma=64)
loglik_ext  = make_log_likelihood(data_ext,  cosmo=cosmo, nx=64, ngamma=64)

# Grid scan over theta and log10_Lj for fixed G0 values
print("\n=== Grid scan: logL_knot(theta, log10_Lj) at G0=10 ===")
thetas = [5, 10, 15, 20, 25, 30]
log_Ljs = [44, 45, 46, 47, 48, 49]
G0_knot = 10.0
G0_ext = 5.0

print(f"{'theta':>8s}", end="")
for lLj in log_Ljs:
    print(f"  {'logLj='+str(lLj):>12s}", end="")
print()

for theta in thetas:
    print(f"{theta:8.1f}", end="")
    for lLj in log_Ljs:
        p = make_params(**FIXED_KNOT, theta=theta, Lj=10.0**lLj, G0=G0_knot)
        ll = float(loglik_knot(p))
        print(f"  {ll:12.1f}", end="")
    print()

print(f"\n=== Grid scan: logL_ext(theta, log10_Lj) at G0={G0_ext} ===")
print(f"{'theta':>8s}", end="")
for lLj in log_Ljs:
    print(f"  {'logLj='+str(lLj):>12s}", end="")
print()

for theta in thetas:
    print(f"{theta:8.1f}", end="")
    for lLj in log_Ljs:
        p = make_params(**FIXED_EXT, theta=theta, Lj=10.0**lLj, G0=G0_ext)
        ll = float(loglik_ext(p))
        print(f"  {ll:12.1f}", end="")
    print()

# Joint logL
print(f"\n=== Grid scan: logL_joint = logL_knot(G0=10) + logL_ext(G0=5) ===")
print(f"{'theta':>8s}", end="")
for lLj in log_Ljs:
    print(f"  {'logLj='+str(lLj):>12s}", end="")
print()

for theta in thetas:
    print(f"{theta:8.1f}", end="")
    for lLj in log_Ljs:
        pk = make_params(**FIXED_KNOT, theta=theta, Lj=10.0**lLj, G0=G0_knot)
        pe = make_params(**FIXED_EXT,  theta=theta, Lj=10.0**lLj, G0=G0_ext)
        ll = float(loglik_knot(pk)) + float(loglik_ext(pe))
        print(f"  {ll:12.1f}", end="")
    print()

# Now scan G0 at fixed theta/Lj
print(f"\n=== Scan G0_knot at theta=12, log10_Lj=47 ===")
for g0 in [2, 5, 8, 10, 12, 15, 20, 25, 30]:
    p = make_params(**FIXED_KNOT, theta=12.0, Lj=10.0**47, G0=float(g0))
    ll = float(loglik_knot(p))
    print(f"  G0={g0:5.1f}  logL_knot = {ll:.2f}")

print(f"\n=== Scan G0_ext at theta=12, log10_Lj=47 ===")
for g0 in [2, 5, 8, 10, 12, 15, 20, 25, 30]:
    p = make_params(**FIXED_EXT, theta=12.0, Lj=10.0**47, G0=float(g0))
    ll = float(loglik_ext(p))
    print(f"  G0={g0:5.1f}  logL_ext  = {ll:.2f}")

# Quick check: what does the SED look like at a reasonable point?
print("\n=== SED at theta=12, log10_Lj=47, G0_knot=10, G0_ext=5 ===")
pk = make_params(**FIXED_KNOT, theta=12.0, Lj=10.0**47, G0=10.0)
pe = make_params(**FIXED_EXT,  theta=12.0, Lj=10.0**47, G0=5.0)

for i, pt in enumerate(data_knot.points):
    nu = jnp.array([pt.nu])
    model_flux = float(observed_sed(nu, pk, cosmo, 64, 64)[0])
    print(f"  Knot ν={pt.nu:.2e}: data={pt.nuFnu:.3e} ± {pt.nuFnu_err:.3e}  model={model_flux:.3e}  "
          f"ratio={model_flux/pt.nuFnu:.2f}")

for i, pt in enumerate(data_ext.points):
    nu = jnp.array([pt.nu])
    model_flux = float(observed_sed(nu, pe, cosmo, 64, 64)[0])
    ul_tag = " [UL]" if pt.is_upper else ""
    print(f"  Ext  ν={pt.nu:.2e}: data={pt.nuFnu:.3e} ± {pt.nuFnu_err:.3e}  model={model_flux:.3e}  "
          f"ratio={model_flux/pt.nuFnu:.2f}{ul_tag}")
