"""
Diagnose why NS fails: check what fraction of the prior volume
has reasonable logL, and test if seeding helps.
"""
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
lo = jnp.array([2.0, 46.0, 46.0, 2.0, 2.0])
hi = jnp.array([30.0, 50.0, 50.0, 30.0, 30.0])

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

loglik_jit = jax.jit(loglikelihood_fn)

# 1. Sample prior volume and check logL distribution
print("=== Prior volume analysis ===")
rng = jax.random.PRNGKey(0)
n_test = 2000
uniform_samples = jax.random.uniform(rng, (n_test, 5), minval=lo, maxval=hi)

lls = []
for i in range(n_test):
    ll = float(loglik_jit(uniform_samples[i]))
    lls.append(ll)
lls = np.array(lls)

print(f"Prior samples: n={n_test}")
print(f"  logL range: [{lls.min():.0f}, {lls.max():.1f}]")
print(f"  logL > -10:  {(lls > -10).sum()} ({(lls > -10).mean()*100:.1f}%)")
print(f"  logL > -50:  {(lls > -50).sum()} ({(lls > -50).mean()*100:.1f}%)")
print(f"  logL > -100: {(lls > -100).sum()} ({(lls > -100).mean()*100:.1f}%)")
print(f"  logL > -500: {(lls > -500).sum()} ({(lls > -500).mean()*100:.1f}%)")
print(f"  median logL: {np.median(lls):.0f}")
print(f"  best logL:   {lls.max():.2f}")
print(f"    at: {uniform_samples[np.argmax(lls)]}")

# 2. What does the MAP seeded population look like?
print(f"\n=== MAP-seeded population (n=500, 20% seeded) ===")
map_x = jnp.array([9.4, 48.22, 48.00, 8.3, 20.1])
n_live = 500

rng, init_key, seed_key = jax.random.split(rng, 3)
n_seeded = n_live // 5  # = 100
n_uniform = n_live - n_seeded  # = 400

uniform = jax.random.uniform(init_key, (n_uniform, 5), minval=lo, maxval=hi)
spread = 0.05 * (hi - lo)
seeded = map_x + spread * jax.random.normal(seed_key, (n_seeded, 5))
seeded = jnp.clip(seeded, lo, hi)

pop = jnp.concatenate([uniform, seeded])

# Evaluate logL for the population
pop_lls = []
for i in range(n_live):
    ll = float(loglik_jit(pop[i]))
    pop_lls.append(ll)
pop_lls = np.array(pop_lls)

print(f"  Uniform part ({n_uniform}): median logL = {np.median(pop_lls[:n_uniform]):.0f}")
print(f"  Seeded part  ({n_seeded}): median logL = {np.median(pop_lls[n_uniform:]):.1f}")
print(f"  Seeded best: {pop_lls[n_uniform:].max():.2f}")
print(f"  Overall: {(pop_lls > -10).sum()} points with logL > -10")
print(f"  Overall: {(pop_lls > -50).sum()} points with logL > -50")

# 3. What if we seed ALL points near MAP?
print(f"\n=== 100% MAP-seeded population ===")
rng, seed_key2 = jax.random.split(rng)
all_seeded = map_x + spread * jax.random.normal(seed_key2, (n_live, 5))
all_seeded = jnp.clip(all_seeded, lo, hi)

seed_lls = []
for i in range(n_live):
    ll = float(loglik_jit(all_seeded[i]))
    seed_lls.append(ll)
seed_lls = np.array(seed_lls)

print(f"  median logL = {np.median(seed_lls):.1f}")
print(f"  best logL = {seed_lls.max():.2f}")
print(f"  worst logL = {seed_lls.min():.1f}")
print(f"  logL > -10: {(seed_lls > -10).sum()} ({(seed_lls > -10).mean()*100:.0f}%)")
print(f"  logL > -50: {(seed_lls > -50).sum()} ({(seed_lls > -50).mean()*100:.0f}%)")

# 4. What about tighter priors?
print(f"\n=== Effect of tighter priors ===")
# Current prior volume
vol_current = float(jnp.prod(hi - lo))
print(f"Current prior volume: {vol_current:.0f}")
print(f"  theta: [{lo[0]:.0f}, {hi[0]:.0f}] range={hi[0]-lo[0]:.0f}")
print(f"  lLj_k: [{lo[1]:.0f}, {hi[1]:.0f}] range={hi[1]-lo[1]:.0f}")
print(f"  lLj_e: [{lo[2]:.0f}, {hi[2]:.0f}] range={hi[2]-lo[2]:.0f}")
print(f"  G0_k:  [{lo[3]:.0f}, {hi[3]:.0f}] range={hi[3]-lo[3]:.0f}")
print(f"  G0_e:  [{lo[4]:.0f}, {hi[4]:.0f}] range={hi[4]-lo[4]:.0f}")

# Tight priors that still contain the MAP
lo_tight = jnp.array([3.0, 47.0, 47.0, 3.0, 5.0])
hi_tight = jnp.array([20.0, 49.5, 49.5, 20.0, 28.0])
vol_tight = float(jnp.prod(hi_tight - lo_tight))
print(f"\nTighter prior volume: {vol_tight:.0f} ({vol_tight/vol_current*100:.1f}% of current)")
print(f"  theta: [{lo_tight[0]:.0f}, {hi_tight[0]:.0f}]")
print(f"  lLj:   [{lo_tight[1]:.0f}, {hi_tight[1]:.1f}]")
print(f"  G0_k:  [{lo_tight[3]:.0f}, {hi_tight[3]:.0f}]")
print(f"  G0_e:  [{lo_tight[4]:.0f}, {hi_tight[4]:.0f}]")

rng, key = jax.random.split(rng)
tight_samples = jax.random.uniform(key, (n_test, 5), minval=lo_tight, maxval=hi_tight)
tight_lls = []
for i in range(n_test):
    ll = float(loglik_jit(tight_samples[i]))
    tight_lls.append(ll)
tight_lls = np.array(tight_lls)
print(f"  Prior samples: median logL = {np.median(tight_lls):.0f}")
print(f"  logL > -10:  {(tight_lls > -10).sum()} ({(tight_lls > -10).mean()*100:.1f}%)")
print(f"  logL > -50:  {(tight_lls > -50).sum()} ({(tight_lls > -50).mean()*100:.1f}%)")
