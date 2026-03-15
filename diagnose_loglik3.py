"""Check if theta < 2 or theta ~ 1 helps, and refine the optimum."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from lumen import (
    make_params, make_cosmology, make_log_likelihood,
    load_sed_Fnu, MODEL_1A, kpc,
)

cosmo = make_cosmology(70.0, 0.3, 0.7)
data_knot = load_sed_Fnu("mydata.csv", unit="mJy")
data_ext = load_sed_Fnu("mydata_ext.csv", unit="mJy")

FIXED_COMMON = dict(q_ratio=1.0, p=2.5, gamma_min=1e2, gamma_max=1e6, z=2.5, eta_e=0.1)
FIXED_KNOT = {**FIXED_COMMON, "Rj": 10*kpc, "l": 10*kpc, "model": int(MODEL_1A)}
FIXED_EXT  = {**FIXED_COMMON, "Rj": 50*kpc, "l": 80*kpc, "model": int(MODEL_1A)}

loglik_knot = make_log_likelihood(data_knot, cosmo=cosmo, nx=64, ngamma=64)
loglik_ext  = make_log_likelihood(data_ext,  cosmo=cosmo, nx=64, ngamma=64)

print("=== Fine 2D scan: theta vs log10_Lj at G0_knot=8, G0_ext=8 ===")
thetas = np.arange(1.0, 8.0, 0.5)
log_Ljs = np.arange(47.5, 50.1, 0.25)

print(f"{'theta':>8s}", end="")
for lLj in log_Ljs:
    print(f"  {f'lLj={lLj:.2f}':>12s}", end="")
print()

best_ll = -1e10
best_params = None

for theta in thetas:
    print(f"{theta:8.1f}", end="")
    for lLj in log_Ljs:
        pk = make_params(**FIXED_KNOT, theta=float(theta), Lj=10.0**float(lLj), G0=8.0)
        pe = make_params(**FIXED_EXT,  theta=float(theta), Lj=10.0**float(lLj), G0=8.0)
        ll = float(loglik_knot(pk)) + float(loglik_ext(pe))
        print(f"  {ll:12.1f}", end="")
        if ll > best_ll:
            best_ll = ll
            best_params = (float(theta), float(lLj))
    print()

print(f"\nBest: theta={best_params[0]:.1f}, log10_Lj={best_params[1]:.2f}, logL={best_ll:.2f}")

# Now fine-tune G0 at the best theta/Lj
theta, lLj = best_params
print(f"\n=== Fine G0 scan at theta={theta:.1f}, log10_Lj={lLj:.2f} ===")
g0_vals = np.arange(2, 16, 0.5)
print(f"{'G0_k':>6s}  {'G0_e':>6s}  {'logL_k':>8s}  {'logL_e':>8s}  {'logL_j':>8s}")

best_ll2 = -1e10
best_g0 = None
for g0k in g0_vals:
    for g0e in g0_vals:
        pk = make_params(**FIXED_KNOT, theta=theta, Lj=10.0**lLj, G0=float(g0k))
        pe = make_params(**FIXED_EXT,  theta=theta, Lj=10.0**lLj, G0=float(g0e))
        ll_k = float(loglik_knot(pk))
        ll_e = float(loglik_ext(pe))
        ll = ll_k + ll_e
        if ll > best_ll2:
            best_ll2 = ll
            best_g0 = (float(g0k), float(g0e), ll_k, ll_e)

print(f"Best G0: knot={best_g0[0]:.1f}, ext={best_g0[1]:.1f}")
print(f"  logL_knot={best_g0[2]:.2f}, logL_ext={best_g0[3]:.2f}, joint={best_ll2:.2f}")

# Final overall best
print(f"\n=== OVERALL BEST ===")
print(f"theta={theta:.1f}, log10_Lj={lLj:.2f}, G0_knot={best_g0[0]:.1f}, G0_ext={best_g0[1]:.1f}")
print(f"logL = {best_ll2:.2f}")
