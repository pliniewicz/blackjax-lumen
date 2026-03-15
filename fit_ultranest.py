#!/usr/bin/env python
"""
Nested sampling for jet SED fitting using UltraNest.

All outputs (plots, samples, summaries) are saved to per-simulation
directories under --output_base. Nothing gets overwritten.

Usage:
    # Single simulation
    python fit_ultranest.py --bank bank2.h5 --sim_idx 42 --use_true_fixed

    # Batch mode — multiple simulations
    python fit_ultranest.py --bank bank2.h5 --use_true_fixed \
        --batch 42 100 1000 5000 10000 50000 100000 154627 200000 300000

    # All 4 models for one source
    python fit_ultranest.py --bank bank2.h5 --sim_idx 42 --use_true_fixed --all_models

    # Real data
    python fit_ultranest.py --data knot_radio.csv --data_xray knot_xray.csv \
        --z 2.5 --Rj_kpc 10 --l_kpc 10 --source_name "4C+19.44_knot"

    # Quick test
    python fit_ultranest.py --bank bank2.h5 --sim_idx 42 --use_true_fixed --quick

    # Collect results from all completed runs into one table
    python fit_ultranest.py --collect results/

Output structure:
    results/
    ├── sim_000042/
    │   ├── summary.json          # machine-readable full results
    │   ├── summary.txt           # human-readable report
    │   ├── corner.pdf            # corner plot (free + derived)
    │   ├── sed_fit.pdf           # SED with posterior draws
    │   ├── sed_components.pdf    # synchrotron + IC separately
    │   ├── marginals.pdf         # 1D marginal posteriors
    │   ├── samples.csv           # posterior samples
    │   └── ultranest_chains/     # UltraNest internal output
    ├── sim_000100/
    │   └── ...
    ├── batch_summary.csv         # one-line-per-sim table
    ├── calibration_recovery.pdf  # true vs recovered
    └── calibration_coverage.pdf  # coverage plot
"""

import argparse
import time
import os
import sys
import numpy as np
import h5py
import json
import csv

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import ultranest
import ultranest.stepsampler

from lumen import (
    make_params, make_cosmology, observed_sed,
    observed_synchrotron, observed_ic,
    kpc, MODEL_NAMES,
)

ALL_MODELS_LIST = [0, 1, 2, 3]


# ==================================================================
#  Prior and likelihood
# ==================================================================

PARAM_DEFS = {
    "G0":       (3.0, 28.0,  "linear"),
    "log10_Lj": (44.0, 50.0, "linear"),
    "theta":    (2.0, 34.0,  "linear"),
}
PARAM_NAMES = list(PARAM_DEFS.keys())
N_PARAMS = len(PARAM_NAMES)


def prior_transform(cube):
    params = np.empty(N_PARAMS)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi, _ = PARAM_DEFS[name]
        params[i] = lo + (hi - lo) * cube[i]
    return params


def make_log_likelihood(nu_obs, flux_obs, flux_err, is_upper,
                        fixed, cosmo, nx=48, ngamma=64,
                        log_err_floor=0.01):
    """
    log_err_floor : float
        Minimum allowed uncertainty in log10-flux space.
        Prevents extremely precise data points from dominating pathologically.
        Default 0.01 (~2.3% in linear flux).  Set to 0.0 to use raw errors.
    """
    nu_jax = jnp.array(nu_obs)
    log_obs = np.log10(np.clip(flux_obs, 1e-300, None))
    log_err = flux_err / (flux_obs * np.log(10.0))
    log_err = np.clip(log_err, log_err_floor, None)

    def _forward(G0, log10_Lj, theta):
        params = make_params(
            G0=G0, q_ratio=1.0, p=fixed["p"], theta=theta,
            gamma_min=fixed["gamma_min"], gamma_max=fixed["gamma_max"],
            Rj=fixed["Rj"], Lj=10.0**log10_Lj, l=fixed["l"],
            z=fixed["z"], eta_e=fixed["eta_e"], model=fixed["model"],
        )
        return observed_sed(nu_jax, params, cosmo, nx=nx, ngamma=ngamma)

    # JIT warmup
    _ = _forward(10.0, 47.0, 12.0).block_until_ready()

    def log_likelihood(params):
        G0, log10_Lj, theta = params
        try:
            model_flux = np.asarray(_forward(G0, log10_Lj, theta))
        except Exception:
            return -1e100
        if np.any(np.isnan(model_flux)) or np.any(model_flux <= 0):
            return -1e100

        log_model = np.log10(np.clip(model_flux, 1e-300, None))
        logl = 0.0
        for i in range(len(nu_obs)):
            if is_upper[i]:
                excess = log_model[i] - log_obs[i]
                if excess > 0:
                    logl -= 0.5 * (excess / log_err[i])**2
            else:
                residual = (log_model[i] - log_obs[i]) / log_err[i]
                logl -= 0.5 * residual**2
        return float(logl)

    return log_likelihood


# ==================================================================
#  Data loading
# ==================================================================

def load_from_bank(h5_path, sim_idx):
    with h5py.File(h5_path, "r") as f:
        slot_freqs = f["slot_frequencies"][:]
        obs_flux = f["obs_flux"][sim_idx]
        obs_err  = f["obs_err"][sim_idx]
        obs_mask = f["obs_mask"][sim_idx]
        p = f["params"]
        true = {
            "G0": float(p["G0"][sim_idx]),
            "log10_Lj": float(np.log10(p["Lj"][sim_idx])),
            "theta": float(p["theta"][sim_idx]),
            "p": float(p["p"][sim_idx]),
            "gamma_min": float(p["gamma_min"][sim_idx]),
            "gamma_max": float(p["gamma_max"][sim_idx]),
            "Rj": float(p["Rj"][sim_idx]),
            "l": float(p["l"][sim_idx]),
            "z": float(p["z"][sim_idx]),
            "eta_e": float(p["eta_e"][sim_idx]),
            "model": int(p["model"][sim_idx]),
        }
    observed = obs_mask != 0
    return (slot_freqs[observed], obs_flux[observed], obs_err[observed],
            obs_mask[observed] < 0, true)


def load_from_csv(csv_path, unit="nuFnu"):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    nu = data[:, 0]
    flux = data[:, 1]
    err = data[:, 2] if data.shape[1] > 2 else 0.1 * flux
    is_ul = data[:, 3].astype(bool) if data.shape[1] > 3 else np.zeros(len(nu), dtype=bool)
    if unit == "mJy":
        nuFnu, nuFnu_err = nu * flux * 1e-26, nu * err * 1e-26
    elif unit == "nJy":
        nuFnu, nuFnu_err = nu * flux * 1e-32, nu * err * 1e-32
    else:
        nuFnu, nuFnu_err = flux, err
    return nu, nuFnu, nuFnu_err, is_ul


# ==================================================================
#  Derived quantities
# ==================================================================

def compute_derived(samples, fixed):
    G0_s, logLj_s, theta_s = samples[:, 0], samples[:, 1], samples[:, 2]
    Lj_s = 10**logLj_s
    Gamma_mid = 1 + (G0_s - 1) * 0.75
    beta = np.sqrt(1 - 1/Gamma_mid**2)
    theta_rad = np.radians(theta_s)
    delta = 1.0 / (Gamma_mid * (1 - beta * np.cos(theta_rad)))
    alpha = (fixed["p"] - 1) / 2
    L_app = delta**(3 + alpha) * Lj_s
    log_L_app = np.log10(np.clip(L_app, 1e-300, None))
    beta_app = beta * np.sin(theta_rad) / (1 - beta * np.cos(theta_rad))
    return {"delta_eff": delta, "log10_L_app": log_L_app, "beta_app": beta_app}


def compute_true_derived(true_vals, fixed):
    G0 = true_vals["G0"]
    Gamma_mid = 1 + (G0 - 1) * 0.75
    beta = np.sqrt(1 - 1/Gamma_mid**2)
    theta_rad = np.radians(true_vals["theta"])
    delta = 1.0 / (Gamma_mid * (1 - beta * np.cos(theta_rad)))
    alpha = (fixed["p"] - 1) / 2
    Lj = 10**true_vals["log10_Lj"]
    L_app = delta**(3 + alpha) * Lj
    beta_app = beta * np.sin(theta_rad) / (1 - beta * np.cos(theta_rad))
    return {"delta_eff": delta, "log10_L_app": np.log10(L_app), "beta_app": beta_app}


# ==================================================================
#  Run nested sampling
# ==================================================================

def run_ultranest(nu_obs, flux_obs, flux_err, is_upper,
                  fixed, cosmo, nx=48, ngamma=64,
                  min_num_live_points=200, max_ncalls=100_000,
                  log_dir=None, resume=True, quick=False,
                  log_err_floor=0.01):
    log_likelihood = make_log_likelihood(
        nu_obs, flux_obs, flux_err, is_upper, fixed, cosmo,
        nx=nx, ngamma=ngamma, log_err_floor=log_err_floor)

    nsteps = 8 if quick else 16
    live = 100 if quick else min_num_live_points

    sampler = ultranest.ReactiveNestedSampler(
        PARAM_NAMES, log_likelihood, prior_transform,
        log_dir=log_dir, resume=resume)
    sampler.stepsampler = ultranest.stepsampler.SliceSampler(
        nsteps=nsteps,
        generate_direction=ultranest.stepsampler.generate_mixture_random_direction)
    result = sampler.run(
        min_num_live_points=live, max_ncalls=max_ncalls,
        show_status=True, viz_callback=False)
    return sampler, result


# ==================================================================
#  Build & save comprehensive output
# ==================================================================

def pct(arr, ps):
    vals = np.percentile(arr, ps)
    return {f"p{p}": float(v) for p, v in zip(ps, vals)}


def build_summary_dict(result, nu_obs, flux_obs, flux_err, is_upper,
                       fixed, true_vals=None, sim_idx=None, elapsed=None):
    samples = result["samples"]
    derived = compute_derived(samples, fixed)
    percentiles = [2.5, 16, 50, 84, 97.5]

    summary = {
        "meta": {
            "sim_idx": sim_idx,
            "n_det": int((~is_upper).sum()),
            "n_ul": int(is_upper.sum()),
            "n_free": N_PARAMS,
            "dof": int((~is_upper).sum()) - N_PARAMS,
            "n_samples": len(samples),
            "ncall": result.get("ncall", None),
            "elapsed_s": elapsed,
        },
        "evidence": {
            "logz": float(result["logz"]),
            "logzerr": float(result["logzerr"]),
        },
        "fixed_params": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                         for k, v in fixed.items()},
        "free_params": {},
        "derived": {},
    }

    for i, name in enumerate(PARAM_NAMES):
        s = samples[:, i]
        entry = {"median": float(np.median(s)), "mean": float(np.mean(s)),
                 "std": float(np.std(s)), **pct(s, percentiles)}
        if true_vals and name in true_vals:
            tv = true_vals[name]
            entry["true"] = float(tv)
            entry["in_68"] = bool(np.percentile(s, 16) <= tv <= np.percentile(s, 84))
            entry["in_95"] = bool(np.percentile(s, 2.5) <= tv <= np.percentile(s, 97.5))
        summary["free_params"][name] = entry

    true_derived = compute_true_derived(true_vals, fixed) if true_vals else {}
    for dname, darr in derived.items():
        entry = {"median": float(np.median(darr)), "mean": float(np.mean(darr)),
                 "std": float(np.std(darr)), **pct(darr, percentiles)}
        if dname in true_derived:
            entry["true"] = float(true_derived[dname])
            entry["in_68"] = bool(np.percentile(darr, 16) <= true_derived[dname] <= np.percentile(darr, 84))
            entry["in_95"] = bool(np.percentile(darr, 2.5) <= true_derived[dname] <= np.percentile(darr, 97.5))
        summary["derived"][dname] = entry

    summary["data"] = {
        "nu_Hz": [float(v) for v in nu_obs],
        "nuFnu": [float(v) for v in flux_obs],
        "nuFnu_err": [float(v) for v in flux_err],
        "is_upper": [bool(v) for v in is_upper],
    }
    if true_vals:
        summary["true_params"] = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                                   for k, v in true_vals.items()}
    return summary


def format_text_summary(summary):
    lines = []
    lines.append("=" * 70)
    lines.append("  ULTRANEST FIT SUMMARY")
    lines.append("=" * 70)
    m = summary["meta"]
    lines.append(f"\n  Simulation index: {m['sim_idx']}")
    lines.append(f"  Data: {m['n_det']} detections, {m['n_ul']} upper limits")
    lines.append(f"  Free parameters: {m['n_free']}, dof: {m['dof']}")
    lines.append(f"  Likelihood evaluations: {m['ncall']}")
    if m["elapsed_s"]:
        lines.append(f"  Wall time: {m['elapsed_s']:.1f}s ({m['elapsed_s']/60:.1f} min)")
    ev = summary["evidence"]
    lines.append(f"\n  log(Z) = {ev['logz']:.3f} ± {ev['logzerr']:.3f}")

    lines.append(f"\n  {'Parameter':>14s}  {'Median':>8s}  {'68% CI':>22s}  "
                 f"{'95% CI':>22s}  {'True':>8s}  {'68%':>3s}  {'95%':>3s}")
    lines.append(f"  {'-' * 90}")

    for section_name, section in [("free_params", summary["free_params"]),
                                   ("derived", summary["derived"])]:
        if section_name == "derived":
            lines.append(f"\n  Derived:")
        for name, vals in section.items():
            true_str = f"{vals['true']:8.3f}" if "true" in vals else "       -"
            c68 = "✓" if vals.get("in_68") else ("✗" if "in_68" in vals else "-")
            c95 = "✓" if vals.get("in_95") else ("✗" if "in_95" in vals else "-")
            lines.append(f"  {name:>14s}  {vals['median']:8.3f}  "
                         f"[{vals['p16']:8.3f}, {vals['p84']:8.3f}]  "
                         f"[{vals['p2.5']:8.3f}, {vals['p97.5']:8.3f}]  "
                         f"{true_str}  {c68:>3s}  {c95:>3s}")

    lines.append(f"\n  Fixed parameters:")
    for k, v in summary["fixed_params"].items():
        lines.append(f"    {k:>12s} = {v:.6g}" if isinstance(v, float) else f"    {k:>12s} = {v}")
    return "\n".join(lines)


def save_all(result, summary, out_dir, nu_obs, flux_obs, flux_err, is_upper,
             fixed, cosmo, true_vals=None,
             flux_lo=1e-19, flux_hi=1e-10, nu_lo=None, nu_hi=None):
    os.makedirs(out_dir, exist_ok=True)

    # 1. JSON
    p = os.path.join(out_dir, "summary.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved {p}")

    # 2. Text
    p = os.path.join(out_dir, "summary.txt")
    with open(p, "w") as f:
        f.write(format_text_summary(summary))
    print(f"  Saved {p}")

    # 3. Samples CSV
    samples = result["samples"]
    derived = compute_derived(samples, fixed)
    csv_path = os.path.join(out_dir, "samples.csv")
    header = list(PARAM_NAMES) + list(derived.keys())
    combined = np.column_stack([samples] + [derived[k] for k in derived])
    np.savetxt(csv_path, combined, delimiter=",", header=",".join(header), comments="")
    print(f"  Saved {csv_path} ({len(samples)} samples)")

    # 4. Plots
    sed_kw = dict(flux_lo=flux_lo, flux_hi=flux_hi, nu_lo=nu_lo, nu_hi=nu_hi)
    import matplotlib
    matplotlib.use("Agg")
    for name, fn, kwargs in [
        ("corner",     _plot_corner, dict(result=result, true_vals=true_vals, fixed=fixed, out_dir=out_dir)),
        ("sed_total",  _plot_sed,    dict(result=result, nu_obs=nu_obs, flux_obs=flux_obs, flux_err=flux_err,
                                          is_upper=is_upper, fixed=fixed, cosmo=cosmo, true_vals=true_vals,
                                          out_dir=out_dir, mode="total", **sed_kw)),
        ("sed_comp",   _plot_sed,    dict(result=result, nu_obs=nu_obs, flux_obs=flux_obs, flux_err=flux_err,
                                          is_upper=is_upper, fixed=fixed, cosmo=cosmo, true_vals=true_vals,
                                          out_dir=out_dir, mode="components", **sed_kw)),
        ("marginals",  _plot_marginals, dict(result=result, true_vals=true_vals, fixed=fixed, out_dir=out_dir)),
    ]:
        try:
            fn(**kwargs)
        except Exception as e:
            print(f"  Warning: {name} plot failed: {e}")


# ==================================================================
#  Plotting
# ==================================================================

def _plot_corner(result, true_vals, fixed, out_dir):
    import matplotlib.pyplot as plt
    samples = result["samples"]
    derived = compute_derived(samples, fixed)
    true_derived = compute_true_derived(true_vals, fixed) if true_vals else {}

    all_names = list(PARAM_NAMES) + ["delta_eff", "log10_L_app", "beta_app"]
    all_labels = [r"$\Gamma_0$", r"log$_{10}(L_j)$", r"$\theta$ [°]",
                  r"$\delta_\mathrm{eff}$", r"log$_{10}(L_\mathrm{app})$", r"$\beta_\mathrm{app}$"]
    all_samples = np.column_stack([
        samples, derived["delta_eff"], derived["log10_L_app"], derived["beta_app"]])

    all_true = {}
    if true_vals:
        for n in PARAM_NAMES:
            if n in true_vals:
                all_true[n] = true_vals[n]
        all_true.update(true_derived)

    N = len(all_names)
    fig, axes = plt.subplots(N, N, figsize=(3 * N, 3 * N))
    for i in range(N):
        for j in range(N):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(all_samples[:, i], bins=60, density=True,
                        color="steelblue", alpha=0.7, edgecolor="none")
                if all_names[i] in all_true:
                    ax.axvline(all_true[all_names[i]], color="red", ls="--", lw=2)
                ax.set_xlabel(all_labels[i], fontsize=10)
                ax.set_yticks([])
            else:
                ax.scatter(all_samples[:, j], all_samples[:, i], s=0.2,
                           alpha=0.1, color="steelblue", rasterized=True)
                if all_names[j] in all_true and all_names[i] in all_true:
                    ax.plot(all_true[all_names[j]], all_true[all_names[i]],
                            "r*", ms=12, mew=1.5, zorder=10)
                if i == N - 1:
                    ax.set_xlabel(all_labels[j], fontsize=10)
                if j == 0:
                    ax.set_ylabel(all_labels[i], fontsize=10)

    fig.suptitle(f"Posterior — log(Z) = {result['logz']:.2f}", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "corner.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_sed(result, nu_obs, flux_obs, flux_err, is_upper,
              fixed, cosmo, true_vals, out_dir, mode="total",
              flux_lo=1e-19, flux_hi=1e-10,
              nu_lo=None, nu_hi=None):
    import matplotlib.pyplot as plt
    samples = result["samples"]
    # Full frequency range: use instrument extremes (LOFAR 54 MHz .. Fermi 2.42e24 Hz)
    # with a factor of 3 padding on each side, unless overridden
    NU_INSTRUMENT_LO = 5.4e7    # LOFAR LBA
    NU_INSTRUMENT_HI = 2.42e24  # Fermi 10 GeV
    log_nu_min = np.log10(nu_lo if nu_lo else NU_INSTRUMENT_LO / 3)
    log_nu_max = np.log10(nu_hi if nu_hi else NU_INSTRUMENT_HI * 3)
    nu_plot = jnp.array(10**np.linspace(log_nu_min, log_nu_max, 300))

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)
    n_draws = min(200, len(samples))
    draw_idx = rng.choice(len(samples), n_draws, replace=False)

    for k, idx in enumerate(draw_idx):
        G0, log10_Lj, theta = samples[idx]
        params = make_params(
            G0=G0, q_ratio=1.0, p=fixed["p"], theta=theta,
            gamma_min=fixed["gamma_min"], gamma_max=fixed["gamma_max"],
            Rj=fixed["Rj"], Lj=10**log10_Lj, l=fixed["l"],
            z=fixed["z"], eta_e=fixed["eta_e"], model=fixed["model"])

        if mode == "total":
            sed = np.asarray(observed_sed(nu_plot, params, cosmo))
            ax.plot(np.asarray(nu_plot), sed, color="steelblue", alpha=0.03, lw=0.5)
        else:
            syn = np.asarray(observed_synchrotron(nu_plot, params, cosmo))
            ic = np.asarray(observed_ic(nu_plot, params, cosmo))
            ax.plot(np.asarray(nu_plot), syn, color="blue", alpha=0.04, lw=0.5,
                    label="Synch" if k == 0 else None)
            ax.plot(np.asarray(nu_plot), ic, color="red", alpha=0.04, lw=0.5,
                    label="IC" if k == 0 else None)

    # Median SED
    med = np.median(samples, axis=0)
    params_med = make_params(
        G0=med[0], q_ratio=1.0, p=fixed["p"], theta=med[2],
        gamma_min=fixed["gamma_min"], gamma_max=fixed["gamma_max"],
        Rj=fixed["Rj"], Lj=10**med[1], l=fixed["l"],
        z=fixed["z"], eta_e=fixed["eta_e"], model=fixed["model"])
    sed_med = np.asarray(observed_sed(nu_plot, params_med, cosmo))
    ax.plot(np.asarray(nu_plot), sed_med, color="black", lw=2, label="Median", zorder=5)

    det = ~is_upper
    if det.any():
        ax.errorbar(nu_obs[det], flux_obs[det], yerr=flux_err[det],
                    fmt="o", ms=7, color="orangered", ecolor="orangered",
                    capsize=3, zorder=10, label="Detections")
    if is_upper.any():
        ax.scatter(nu_obs[is_upper], flux_obs[is_upper], marker="v", s=60,
                   color="gray", edgecolors="k", linewidths=0.5, zorder=10, label="UL")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$ [Hz]", fontsize=13)
    ax.set_ylabel(r"$\nu F_\nu$ [erg/s/cm²]", fontsize=13)

    # Axis limits: match the physically relevant flux range
    ax.set_ylim(flux_lo * 0.3, flux_hi * 5)
    ax.axhline(flux_lo, color="green", ls=":", lw=0.7, alpha=0.5)
    ax.axhline(flux_hi, color="green", ls=":", lw=0.7, alpha=0.5)

    ax.set_title("Total SED" if mode == "total" else "Synch + IC components", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    fname = "sed_fit.pdf" if mode == "total" else "sed_components.pdf"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_marginals(result, true_vals, fixed, out_dir):
    import matplotlib.pyplot as plt
    samples = result["samples"]
    derived = compute_derived(samples, fixed)
    true_derived = compute_true_derived(true_vals, fixed) if true_vals else {}

    names = list(PARAM_NAMES) + ["delta_eff", "log10_L_app"]
    labels = [r"$\Gamma_0$", r"log$_{10}(L_j)$", r"$\theta$ [°]",
              r"$\delta_\mathrm{eff}$", r"log$_{10}(L_\mathrm{app})$"]
    arrays = [samples[:, 0], samples[:, 1], samples[:, 2],
              derived["delta_eff"], derived["log10_L_app"]]

    all_true = {}
    if true_vals:
        for n in PARAM_NAMES:
            if n in true_vals: all_true[n] = true_vals[n]
        all_true.update(true_derived)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5))
    for ax, name, label, arr in zip(axes, names, labels, arrays):
        ax.hist(arr, bins=60, density=True, color="steelblue", alpha=0.7, edgecolor="none")
        lo, hi = np.percentile(arr, [16, 84])
        ax.axvspan(lo, hi, alpha=0.15, color="steelblue", label="68% CI")
        ax.axvline(np.median(arr), color="steelblue", ls="-", lw=1.5)
        if name in all_true:
            ax.axvline(all_true[name], color="red", ls="--", lw=2, label="True")
        ax.set_xlabel(label, fontsize=12)
        ax.set_yticks([])
        ax.legend(fontsize=8)

    fig.suptitle(f"Marginal posteriors — log(Z) = {result['logz']:.2f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "marginals.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ==================================================================
#  Model comparison
# ==================================================================

def compare_models(nu_obs, flux_obs, flux_err, is_upper, fixed_base, cosmo,
                   out_dir, true_model=None, nx=48, ngamma=64,
                   min_num_live_points=200, max_ncalls=80_000, quick=False,
                   log_err_floor=0.01):
    results, evidences = {}, {}

    for model_id in ALL_MODELS_LIST:
        model_name = MODEL_NAMES.get(model_id, str(model_id))
        print(f"\n{'#'*70}\n  Model {model_name}\n{'#'*70}")

        fixed = dict(fixed_base); fixed["model"] = model_id
        log_dir = os.path.join(out_dir, f"ultranest_{model_name}")

        sampler, result = run_ultranest(
            nu_obs, flux_obs, flux_err, is_upper, fixed, cosmo,
            nx=nx, ngamma=ngamma, min_num_live_points=min_num_live_points,
            max_ncalls=max_ncalls, log_dir=log_dir, quick=quick,
            log_err_floor=log_err_floor)

        results[model_id] = result
        evidences[model_id] = (result["logz"], result["logzerr"])

    max_logz = max(e[0] for e in evidences.values())
    sorted_models = sorted(evidences.keys(), key=lambda m: -evidences[m][0])

    lines = [f"\n{'='*70}", "  MODEL COMPARISON (Bayesian Evidence)", f"{'='*70}",
             f"\n  {'Model':>8s}  {'log(Z)':>10s}  {'±':>6s}  {'Δlog(Z)':>10s}  "
             f"{'Bayes factor':>14s}  {'Interpretation':>20s}",
             f"  {'-'*80}"]

    model_results = []
    for model_id in sorted_models:
        logz, logz_err = evidences[model_id]
        dlz = logz - max_logz
        bf = np.exp(dlz)
        name = MODEL_NAMES.get(model_id, str(model_id))
        interp = ("Not worth mentioning" if abs(dlz) < 1 else
                  "Substantial" if abs(dlz) < 2.5 else
                  "Strong" if abs(dlz) < 5 else "Decisive")
        true_tag = " ← true" if model_id == true_model else ""
        best_tag = " ★" if dlz == 0 else ""
        lines.append(f"  {name:>8s}  {logz:10.2f}  {logz_err:6.2f}  {dlz:+10.2f}  "
                     f"{bf:14.4f}  {interp:>20s}{true_tag}{best_tag}")
        model_results.append({"model": name, "model_id": model_id,
                              "logz": logz, "logzerr": logz_err,
                              "delta_logz": dlz, "bayes_factor": bf,
                              "interpretation": interp, "is_true": model_id == true_model})

    logzs = np.array([evidences[m][0] for m in sorted_models])
    probs = np.exp(logzs - logzs.max()) / np.sum(np.exp(logzs - logzs.max()))
    lines.append(f"\n  Posterior model probabilities (equal prior):")
    for i, (mid, prob) in enumerate(zip(sorted_models, probs)):
        name = MODEL_NAMES.get(mid, str(mid))
        true_tag = " ← true" if mid == true_model else ""
        lines.append(f"    {name:>6s}: {prob:6.1%}  {'█' * int(prob * 40)}{true_tag}")
        model_results[i]["probability"] = float(prob)

    report = "\n".join(lines)
    print(report)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model_comparison.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(out_dir, "model_comparison.json"), "w") as f:
        json.dump(model_results, f, indent=2)
    print(f"  Saved model_comparison.* to {out_dir}")

    return results, evidences


# ==================================================================
#  Batch summary collector
# ==================================================================

def collect_results(base_dir):
    import glob

    json_files = sorted(glob.glob(os.path.join(base_dir, "sim_*/summary.json")) +
                        glob.glob(os.path.join(base_dir, "*/summary.json")))
    # Deduplicate
    json_files = sorted(set(json_files))

    if not json_files:
        print(f"  No summary.json found in {base_dir}")
        return

    print(f"  Found {len(json_files)} completed runs")
    rows = []

    for jf in json_files:
        with open(jf) as f:
            s = json.load(f)
        row = {"sim_idx": s["meta"]["sim_idx"],
               "n_det": s["meta"]["n_det"], "n_ul": s["meta"]["n_ul"],
               "ncall": s["meta"]["ncall"], "elapsed_s": s["meta"]["elapsed_s"],
               "logz": s["evidence"]["logz"], "logzerr": s["evidence"]["logzerr"]}
        for name in PARAM_NAMES:
            p = s["free_params"][name]
            row[f"{name}_med"] = p["median"]
            row[f"{name}_p16"] = p["p16"]
            row[f"{name}_p84"] = p["p84"]
            if "true" in p:
                row[f"{name}_true"] = p["true"]
                row[f"{name}_in68"] = p["in_68"]
                row[f"{name}_in95"] = p["in_95"]
        for dname in ["delta_eff", "log10_L_app"]:
            if dname in s["derived"]:
                d = s["derived"][dname]
                row[f"{dname}_med"] = d["median"]
                row[f"{dname}_p16"] = d["p16"]
                row[f"{dname}_p84"] = d["p84"]
                if "true" in d:
                    row[f"{dname}_true"] = d["true"]
                    row[f"{dname}_in68"] = d["in_68"]
                    row[f"{dname}_in95"] = d["in_95"]
        rows.append(row)

    # Save CSV
    csv_path = os.path.join(base_dir, "batch_summary.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved {csv_path}")

    # Coverage table
    has_truth = "G0_true" in rows[0]
    if has_truth:
        print(f"\n  {'='*70}")
        print(f"  CALIBRATION ({len(rows)} simulations)")
        print(f"  {'='*70}")
        print(f"  {'Parameter':>14s}  {'68% coverage':>14s}  {'95% coverage':>14s}")
        print(f"  {'-'*48}")
        for name in list(PARAM_NAMES) + ["delta_eff", "log10_L_app"]:
            k68, k95 = f"{name}_in68", f"{name}_in95"
            v68 = [r[k68] for r in rows if k68 in r]
            v95 = [r[k95] for r in rows if k95 in r]
            if v68:
                c68 = sum(v68) / len(v68)
                c95 = sum(v95) / len(v95)
                ok68 = "✓" if abs(c68 - 0.68) < 0.15 else "⚠"
                ok95 = "✓" if abs(c95 - 0.95) < 0.10 else "⚠"
                print(f"  {name:>14s}  {c68:8.0%} {ok68:>5s}  {c95:8.0%} {ok95:>5s}")

    # Calibration plots
    if has_truth and len(rows) >= 3:
        try:
            _plot_calibration(rows, base_dir)
        except Exception as e:
            print(f"  Calibration plots failed: {e}")

    return rows


def _plot_calibration(rows, base_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params_to_plot = ["G0", "log10_Lj", "theta", "log10_L_app"]
    labels = [r"$\Gamma_0$", r"log$_{10}(L_j)$", r"$\theta$ [°]", r"log$_{10}(L_\mathrm{app})$"]

    # Recovery plot
    fig, axes = plt.subplots(1, len(params_to_plot), figsize=(5 * len(params_to_plot), 5))
    for ax, name, label in zip(axes, params_to_plot, labels):
        tk, mk, lk, hk = f"{name}_true", f"{name}_med", f"{name}_p16", f"{name}_p84"
        valid = [r for r in rows if tk in r and mk in r]
        if not valid:
            ax.set_visible(False); continue
        trues = np.array([r[tk] for r in valid])
        meds = np.array([r[mk] for r in valid])
        los = np.array([r[lk] for r in valid])
        his = np.array([r[hk] for r in valid])
        errs = np.array([meds - los, his - meds])
        ax.errorbar(trues, meds, yerr=errs, fmt="o", ms=6, color="steelblue",
                    ecolor="steelblue", capsize=3, alpha=0.8)
        lo_lim = min(trues.min(), (meds - errs[0]).min()) * 0.95
        hi_lim = max(trues.max(), (meds + errs[1]).max()) * 1.05
        if name == "log10_Lj" or name == "log10_L_app":
            lo_lim -= 0.5; hi_lim += 0.5
        ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "k--", lw=1, alpha=0.5)
        ax.set_xlim(lo_lim, hi_lim); ax.set_ylim(lo_lim, hi_lim)
        ax.set_xlabel(f"True {label}", fontsize=12)
        ax.set_ylabel(f"Recovered {label}", fontsize=12)
        ax.set_aspect("equal")

    fig.suptitle(f"Parameter Recovery ({len(rows)} sims)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(base_dir, "calibration_recovery.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Coverage plot
    fig, axes = plt.subplots(1, len(params_to_plot), figsize=(5 * len(params_to_plot), 5))
    for ax, name, label in zip(axes, params_to_plot, labels):
        k68, k95 = f"{name}_in68", f"{name}_in95"
        v68 = [r[k68] for r in rows if k68 in r]
        v95 = [r[k95] for r in rows if k95 in r]
        if not v68:
            ax.set_visible(False); continue
        cov68 = sum(v68) / len(v68)
        cov95 = sum(v95) / len(v95)

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Ideal")
        ax.scatter([0.68, 0.95], [cov68, cov95], s=120, color="steelblue",
                   edgecolors="k", zorder=5)
        ax.annotate(f"{cov68:.0%}", xy=(0.68, cov68), fontsize=11, fontweight="bold",
                    xytext=(0.15, max(cov68 + 0.1, 0.85)),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.annotate(f"{cov95:.0%}", xy=(0.95, cov95), fontsize=11, fontweight="bold",
                    xytext=(0.5, min(cov95 - 0.15, 0.3)),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.set_xlabel("Nominal coverage", fontsize=12)
        ax.set_ylabel("Empirical coverage", fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    fig.suptitle(f"Coverage Calibration ({len(rows)} sims)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(base_dir, "calibration_coverage.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ==================================================================
#  Main
# ==================================================================

def run_single(sim_idx, args, nu_obs, flux_obs, flux_err, is_upper,
               fixed, cosmo, true_vals, output_base):
    if sim_idx is not None:
        out_dir = os.path.join(output_base, f"sim_{sim_idx:06d}")
    elif hasattr(args, 'source_name') and args.source_name:
        out_dir = os.path.join(output_base, args.source_name)
    else:
        out_dir = os.path.join(output_base, "fit")

    log_dir = os.path.join(out_dir, "ultranest_chains")

    print(f"\n{'='*70}")
    print(f"  SIM #{sim_idx}  →  {out_dir}")
    print(f"  {(~is_upper).sum()} det, {is_upper.sum()} UL")
    if true_vals:
        mn = MODEL_NAMES.get(true_vals.get('model'), '?')
        print(f"  True: G0={true_vals['G0']:.2f}  log10Lj={true_vals['log10_Lj']:.3f}  "
              f"theta={true_vals['theta']:.2f}°  model={mn}")
    print(f"{'='*70}")

    t0 = time.perf_counter()

    if args.all_models:
        fixed_base = {k: v for k, v in fixed.items() if k != "model"}
        compare_models(
            nu_obs, flux_obs, flux_err, is_upper, fixed_base, cosmo,
            out_dir=out_dir,
            true_model=true_vals["model"] if true_vals else None,
            nx=args.nx, ngamma=args.ngamma,
            min_num_live_points=args.live_points,
            max_ncalls=args.max_ncalls, quick=args.quick,
            log_err_floor=args.log_err_floor)
        elapsed = time.perf_counter() - t0
    else:
        sampler, result = run_ultranest(
            nu_obs, flux_obs, flux_err, is_upper, fixed, cosmo,
            nx=args.nx, ngamma=args.ngamma,
            min_num_live_points=args.live_points,
            max_ncalls=args.max_ncalls, log_dir=log_dir, quick=args.quick,
            log_err_floor=args.log_err_floor)

        elapsed = time.perf_counter() - t0

        summary = build_summary_dict(
            result, nu_obs, flux_obs, flux_err, is_upper, fixed,
            true_vals=true_vals, sim_idx=sim_idx, elapsed=elapsed)

        print(format_text_summary(summary))

        save_all(result, summary, out_dir,
                 nu_obs, flux_obs, flux_err, is_upper,
                 fixed, cosmo, true_vals=true_vals,
                 flux_lo=args.flux_lo, flux_hi=args.flux_hi,
                 nu_lo=args.nu_lo, nu_hi=args.nu_hi)

    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="UltraNest SED fitting")
    parser.add_argument("--bank", help="HDF5 simulation bank")
    parser.add_argument("--sim_idx", type=int, default=None)
    parser.add_argument("--data", help="CSV data file")
    parser.add_argument("--data_xray", help="Additional X-ray CSV")
    parser.add_argument("--unit", default="nuFnu")
    parser.add_argument("--source_name", default=None)

    parser.add_argument("--z", type=float, default=None)
    parser.add_argument("--Rj_kpc", type=float, default=10.0)
    parser.add_argument("--l_kpc", type=float, default=10.0)
    parser.add_argument("--p_val", type=float, default=2.5)
    parser.add_argument("--gamma_min", type=float, default=1e2)
    parser.add_argument("--gamma_max", type=float, default=1e6)
    parser.add_argument("--eta_e", type=float, default=0.1)
    parser.add_argument("--model", type=int, default=0)
    parser.add_argument("--use_true_fixed", action="store_true")

    parser.add_argument("--live_points", type=int, default=400)
    parser.add_argument("--max_ncalls", type=int, default=100_000)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ngamma", type=int, default=64)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--log_err_floor", type=float, default=0.01,
                        help="Min uncertainty in log10-flux space (default 0.01 ≈ 2.3%% linear). "
                             "Set to 0.0 to use raw propagated errors.")

    parser.add_argument("--output_base", default="results")
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--batch", nargs="+", type=int, default=None)
    parser.add_argument("--collect", nargs="?", const="results",
                        help="Collect results into summary table + calibration plots")

    parser.add_argument("--flux_lo", type=float, default=1e-19,
                        help="Lower flux limit for SED plots [erg/s/cm²] (default 1e-19)")
    parser.add_argument("--flux_hi", type=float, default=1e-10,
                        help="Upper flux limit for SED plots [erg/s/cm²] (default 1e-10)")
    parser.add_argument("--nu_lo", type=float, default=None,
                        help="Lower frequency limit for SED plots [Hz] "
                             "(default: LOFAR LBA / 3 ≈ 1.8e7 Hz)")
    parser.add_argument("--nu_hi", type=float, default=None,
                        help="Upper frequency limit for SED plots [Hz] "
                             "(default: Fermi 10 GeV × 3 ≈ 7.3e24 Hz)")

    args = parser.parse_args()

    # --- Collect mode ---
    if args.collect:
        collect_results(args.collect)
        return

    # --- Determine sim indices ---
    if args.batch:
        sim_indices = args.batch
    elif args.sim_idx is not None:
        sim_indices = [args.sim_idx]
    elif args.data:
        sim_indices = [None]
    else:
        parser.error("Provide --sim_idx, --batch, --data, or --collect")

    print("Warming up JIT...")
    t_total = time.perf_counter()

    for k, sim_idx in enumerate(sim_indices):
        true_vals = None

        if args.bank:
            nu_obs, flux_obs, flux_err, is_upper, true_vals = load_from_bank(args.bank, sim_idx)

            if args.use_true_fixed:
                fixed = {key: true_vals[key] for key in
                         ["p", "gamma_min", "gamma_max", "Rj", "l", "z", "eta_e", "model"]}
            else:
                z = args.z if args.z is not None else true_vals["z"]
                fixed = {"p": args.p_val, "gamma_min": args.gamma_min,
                         "gamma_max": args.gamma_max, "Rj": args.Rj_kpc * kpc,
                         "l": args.l_kpc * kpc, "z": z,
                         "eta_e": args.eta_e, "model": args.model}

        elif args.data:
            nu_obs, flux_obs, flux_err, is_upper = load_from_csv(args.data, args.unit)
            if args.data_xray:
                nu_x, fx_x, err_x, ul_x = load_from_csv(args.data_xray, "nJy")
                nu_obs = np.concatenate([nu_obs, nu_x])
                flux_obs = np.concatenate([flux_obs, fx_x])
                flux_err = np.concatenate([flux_err, err_x])
                is_upper = np.concatenate([is_upper, ul_x])
            if args.z is None:
                parser.error("--z is required for CSV data")
            fixed = {"p": args.p_val, "gamma_min": args.gamma_min,
                     "gamma_max": args.gamma_max, "Rj": args.Rj_kpc * kpc,
                     "l": args.l_kpc * kpc, "z": args.z,
                     "eta_e": args.eta_e, "model": args.model}

        cosmo = make_cosmology(71.0, 0.27, 0.73)
        run_single(sim_idx, args, nu_obs, flux_obs, flux_err, is_upper,
                   fixed, cosmo, true_vals, args.output_base)

        if len(sim_indices) > 1:
            print(f"\n  === [{k+1}/{len(sim_indices)}] completed ===\n")

    # Auto-collect after batch
    if len(sim_indices) > 1:
        print(f"\n{'#'*70}")
        print(f"  BATCH COMPLETE — collecting results")
        print(f"{'#'*70}")
        collect_results(args.output_base)

    print(f"\n  Grand total: {(time.perf_counter() - t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
