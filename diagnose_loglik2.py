"""Fine grid scan near the best region found in the coarse scan."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from lumen import (
    make_params, make_cosmology, make_log_likelihood,
    load_sed_Fnu, observed_sed,
    MODEL_1A, kpc,
)

cosmo = make_cosmology(70.0, 0.3, 0.7)
data_knot = load_sed_Fnu("mydata.csv", unit="mJy")
data_ext = load_sed_Fnu("mydata_ext.csv", unit="mJy")

FIXED_COMMON = dict(q_ratio=1.0, p=2.5, gamma_min=1e2, gamma_max=1e6, z=2.5, eta_e=0.1)
FIXED_KNOT = {**FIXED_COMMON, "Rj": 10*kpc, "l": 10*kpc, "model": int(MODEL_1A)}
FIXED_EXT  = {**FIXED_COMMON, "Rj": 50*kpc, "l": 80*kpc, "model": int(MODEL_1A)}

loglik_knot = make_log_likelihood(data_knot, cosmo=cosmo, nx=64, ngamma=64)
loglik_ext  = make_log_likelihood(data_ext,  cosmo=cosmo, nx=64, ngamma=64)

# Fine scan near the best region
print("=== Fine scan: joint logL, G0_knot=2, G0_ext=2 ===")
thetas = np.arange(2, 12, 1.0)
log_Ljs = np.arange(48.5, 50.1, 0.25)

print(f"{'theta':>8s}", end="")
for lLj in log_Ljs:
    print(f"  {'lLj='+f'{lLj:.2f}':>12s}", end="")
print()

best_ll = -1e10
best_params = None

for theta in thetas:
    print(f"{theta:8.1f}", end="")
    for lLj in log_Ljs:
        for g0k in [2, 3, 5]:
            for g0e in [2, 3, 5]:
                pk = make_params(**FIXED_KNOT, theta=float(theta), Lj=10.0**float(lLj), G0=float(g0k))
                pe = make_params(**FIXED_EXT,  theta=float(theta), Lj=10.0**float(lLj), G0=float(g0e))
                ll_k = float(loglik_knot(pk))
                ll_e = float(loglik_ext(pe))
                ll = ll_k + ll_e
                if ll > best_ll:
                    best_ll = ll
                    best_params = (float(theta), float(lLj), float(g0k), float(g0e), ll_k, ll_e)
        # Print only the best G0 combo for this theta/lLj
        pk = make_params(**FIXED_KNOT, theta=float(theta), Lj=10.0**float(lLj), G0=float(best_params[2]) if best_params[1]==float(lLj) and best_params[0]==float(theta) else 2.0)
        pe = make_params(**FIXED_EXT,  theta=float(theta), Lj=10.0**float(lLj), G0=float(best_params[3]) if best_params[1]==float(lLj) and best_params[0]==float(theta) else 2.0)
        ll = float(loglik_knot(pk)) + float(loglik_ext(pe))
        print(f"  {ll:12.1f}", end="")
    print()

print(f"\nBest found: theta={best_params[0]:.1f}, log10Lj={best_params[1]:.2f}, "
      f"G0_knot={best_params[2]:.1f}, G0_ext={best_params[3]:.1f}")
print(f"  logL_knot={best_params[4]:.2f}, logL_ext={best_params[5]:.2f}, "
      f"logL_joint={best_ll:.2f}")

# Show SED at this best point
print(f"\n=== SED at best point ===")
theta, lLj, g0k, g0e = best_params[:4]
pk = make_params(**FIXED_KNOT, theta=theta, Lj=10.0**lLj, G0=g0k)
pe = make_params(**FIXED_EXT,  theta=theta, Lj=10.0**lLj, G0=g0e)

for i, pt in enumerate(data_knot.points):
    nu = jnp.array([pt.nu])
    model_flux = float(observed_sed(nu, pk, cosmo, 64, 64)[0])
    log_data = np.log10(pt.nuFnu)
    log_model = np.log10(max(model_flux, 1e-300))
    print(f"  Knot ν={pt.nu:.2e}: log10(data)={log_data:.3f}  log10(model)={log_model:.3f}  "
          f"residual={log_data-log_model:.3f} dex")

for i, pt in enumerate(data_ext.points):
    nu = jnp.array([pt.nu])
    model_flux = float(observed_sed(nu, pe, cosmo, 64, 64)[0])
    log_data = np.log10(pt.nuFnu)
    log_model = np.log10(max(model_flux, 1e-300))
    ul_tag = " [UL]" if pt.is_upper else ""
    print(f"  Ext  ν={pt.nu:.2e}: log10(data)={log_data:.3f}  log10(model)={log_model:.3f}  "
          f"residual={log_data-log_model:.3f} dex{ul_tag}")

# Also scan with a broader G0 range at the best theta/Lj
print(f"\n=== G0 scan at theta={theta:.0f}, log10Lj={lLj:.2f} ===")
print(f"  {'G0_k':>6s}  {'G0_e':>6s}  {'logL_knot':>10s}  {'logL_ext':>10s}  {'logL_joint':>11s}")
for g0k in [2, 3, 4, 5, 8, 10, 15]:
    for g0e in [2, 3, 4, 5, 8, 10, 15]:
        pk = make_params(**FIXED_KNOT, theta=theta, Lj=10.0**lLj, G0=float(g0k))
        pe = make_params(**FIXED_EXT,  theta=theta, Lj=10.0**lLj, G0=float(g0e))
        ll_k = float(loglik_knot(pk))
        ll_e = float(loglik_ext(pe))
        if ll_k + ll_e > best_ll - 20:  # Only print near-optimal combos
            print(f"  {g0k:6.1f}  {g0e:6.1f}  {ll_k:10.2f}  {ll_e:10.2f}  {ll_k+ll_e:11.2f}")
