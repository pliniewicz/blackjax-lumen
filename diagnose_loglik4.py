"""
Diagnostic with corrected physical parameters from arXiv:1509.04822.
Cosmology: H0=71, Om=0.27, OL=0.73
Knot geometry: Rj=1.5 kpc, l=3 kpc (3x3 kpc under 1 beam)
Extended: Rj=3 kpc, l=80 kpc  (maybe broader, but paper value)
gamma_min=10, gamma_max=1e5
Expected: theta~7.5 deg, G0~10, Lj~(0.3-3)e47 => log10_Lj~46.5-47.5
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from lumen import (
    make_params, make_cosmology, make_log_likelihood,
    load_sed_Fnu, observed_sed,
    MODEL_1A, MODEL_1B, MODEL_2A, MODEL_2B, MODEL_NAMES, kpc,
)

cosmo = make_cosmology(71.0, 0.27, 0.73)
data_knot = load_sed_Fnu("mydata.csv", unit="mJy")
data_ext = load_sed_Fnu("mydata_ext.csv", unit="mJy")

FIXED_COMMON = dict(q_ratio=1.0, p=2.5, gamma_min=10.0, gamma_max=1e5, z=2.5, eta_e=0.1)
FIXED_KNOT = {**FIXED_COMMON, "Rj": 1.5*kpc, "l": 3*kpc}
FIXED_EXT  = {**FIXED_COMMON, "Rj": 3*kpc,   "l": 80*kpc}

for model_id, model_name in [(MODEL_1A, "MODEL_1A"), (MODEL_1B, "MODEL_1B"),
                               (MODEL_2A, "MODEL_2A"), (MODEL_2B, "MODEL_2B")]:
    FIXED_KNOT_M = {**FIXED_KNOT, "model": int(model_id)}
    FIXED_EXT_M  = {**FIXED_EXT,  "model": int(model_id)}

    loglik_knot = make_log_likelihood(data_knot, cosmo=cosmo, nx=64, ngamma=64)
    loglik_ext  = make_log_likelihood(data_ext,  cosmo=cosmo, nx=64, ngamma=64)

    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"{'='*70}")

    # Scan around expected values
    thetas = np.array([3, 5, 7.5, 10, 12, 15, 20])
    log_Ljs = np.array([45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5])

    print(f"\n  Joint logL scan (G0_knot=10, G0_ext=10):")
    print(f"  {'theta':>8s}", end="")
    for lLj in log_Ljs:
        print(f"  {f'lLj={lLj:.1f}':>11s}", end="")
    print()

    best_ll = -1e10
    best_pt = None

    for theta in thetas:
        print(f"  {theta:8.1f}", end="")
        for lLj in log_Ljs:
            pk = make_params(**FIXED_KNOT_M, theta=float(theta), Lj=10.0**float(lLj), G0=10.0)
            pe = make_params(**FIXED_EXT_M,  theta=float(theta), Lj=10.0**float(lLj), G0=10.0)
            ll = float(loglik_knot(pk)) + float(loglik_ext(pe))
            print(f"  {ll:11.1f}", end="")
            if ll > best_ll:
                best_ll = ll
                best_pt = (float(theta), float(lLj))
        print()

    # Fine G0 scan at best theta/Lj
    theta, lLj = best_pt
    best_ll2 = -1e10
    best_g0 = None
    for g0k in np.arange(3, 20, 1.0):
        for g0e in np.arange(3, 20, 1.0):
            pk = make_params(**FIXED_KNOT_M, theta=theta, Lj=10.0**lLj, G0=float(g0k))
            pe = make_params(**FIXED_EXT_M,  theta=theta, Lj=10.0**lLj, G0=float(g0e))
            ll = float(loglik_knot(pk)) + float(loglik_ext(pe))
            if ll > best_ll2:
                best_ll2 = ll
                best_g0 = (float(g0k), float(g0e))

    print(f"\n  Best: theta={theta:.1f}, log10_Lj={lLj:.1f}, "
          f"G0_knot={best_g0[0]:.0f}, G0_ext={best_g0[1]:.0f}, logL={best_ll2:.2f}")

    # Show SED residuals at best point
    pk = make_params(**FIXED_KNOT_M, theta=theta, Lj=10.0**lLj, G0=best_g0[0])
    pe = make_params(**FIXED_EXT_M,  theta=theta, Lj=10.0**lLj, G0=best_g0[1])
    print(f"  SED residuals:")
    for pt in data_knot.points:
        nu = jnp.array([pt.nu])
        mf = float(observed_sed(nu, pk, cosmo, 64, 64)[0])
        print(f"    Knot ν={pt.nu:.2e}: data={pt.nuFnu:.2e}  model={mf:.2e}  "
              f"ratio={mf/pt.nuFnu:.3f}")
    for pt in data_ext.points:
        nu = jnp.array([pt.nu])
        mf = float(observed_sed(nu, pe, cosmo, 64, 64)[0])
        ul = " [UL]" if pt.is_upper else ""
        print(f"    Ext  ν={pt.nu:.2e}: data={pt.nuFnu:.2e}  model={mf:.2e}  "
              f"ratio={mf/pt.nuFnu:.3f}{ul}")
