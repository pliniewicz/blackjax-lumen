"""
Lumen — Astrophysical jet SED modelling in JAX.

Synchrotron-only skeleton for validation against the Julia implementation.
IC/CMB emission follows the same architecture and will be added next.
"""

# Enable 64-bit precision globally (critical for astrophysical calculations)
import jax
jax.config.update("jax_enable_x64", True)

# --- Types & models ---
from .types import (
    JetParams, make_params,
    MODEL_1A, MODEL_1B, MODEL_2A, MODEL_2B,
    MODEL_NAMES,
)

# --- Physical constants (CGS-Gaussian) ---
from .constants import (
    c, q, h, hbar, me, r0, kB, sigma_T,
    T_CMB, U_CMB, eps_CMB, pc, kpc, LCONST,
    c_km_s, Mpc_cm,
)

# --- Integration ---
from .integration import get_gl, GL_N, gl_quad, map_to_interval

# --- Jet profiles ---
from .profiles import (
    Gamma_profile, b_profile, f_profile, p_profile,
    lorentz_beta, doppler, Doppler_factor,
    PB1, B1,
)

# --- Electrons ---
from .electrons import (
    simple_electron_distribution, electron_distribution,
    electron_norm, K_e,
)

# --- Synchrotron ---
from .synchrotron import synchR, synchrotron_spectrum, synchrotron_luminosity

# --- Inverse Compton ---
from .ic import ic_kernel, ic_spectrum, ic_luminosity

# --- Cosmology ---
from .cosmology import (
    Cosmology, make_cosmology, Planck18, WMAP9,
    E_z, hubble_distance, comoving_distance,
    transverse_comoving_distance, angular_diameter_distance,
    luminosity_distance, luminosity_distance_cm,
    kpc_per_arcsec,
    nuFnu_to_nuLnu, nuLnu_to_nuFnu,
    Fnu_to_nuFnu,
)

# --- Observed SED ---
from .sed import observed_synchrotron, observed_ic, observed_sed

# --- Data I/O & fitting ---
from .fitting import (
    SEDDataPoint, SEDData,
    load_sed, load_sed_Fnu,
    log_likelihood, make_log_likelihood,
)
