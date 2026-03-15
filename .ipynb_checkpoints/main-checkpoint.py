import numpy as np
import h5py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from matplotlib import pyplot as plt

from lumen import (
    make_params, make_cosmology, observed_sed,
    MODEL_1A, MODEL_1B, MODEL_2A, MODEL_2B, 
    kpc, MODEL_NAMES,
    load_sed, load_sed_Fnu,
    log_likelihood
)

cosmo = make_cosmology(70, 0.3, 0.7)

p = make_params(
        G0=10.0, q_ratio=1.0, p=2.5, theta=12.0,
        gamma_min=1e2, gamma_max=1e6,
        Rj=10*kpc, Lj=1e48, l=10*kpc,
        z=2.5, eta_e=0.1, model=int(MODEL_1A),
)

# nu_jax = jnp.array(10**np.arange(3, 26, 0.01))
#
# dupa = observed_sed(nu_jax, p, cosmo, nx=128, ngamma=128)
#
# fig, ax = plt.subplots(figsize=(10, 6))
#
# ax.plot(nu_jax, dupa, '-', color='0.00', lw=1.5, zorder=1,
#        label='Test')
#
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel(r'$\nu$ [Hz]', fontsize=13)
# ax.set_ylabel(r'$\nu F_\nu$ [erg/s/cm$^2$]', fontsize=13)
#
# ax.set_xlim(1e3, 1e26)
# ax.set_ylim(1e-24, 1e-14)


# fig.tight_layout()
#
# fig.savefig('dupa.pdf', dpi=180)
#
# print(len(nu_jax))

data = load_sed_Fnu("mydata.csv", unit='mJy')
#
# ll = log_likelihood(
#       p,
#       data=data,
#       cosmo=cosmo,
#   )
#
# print(ll)

print("nu:", data.frequencies)                                                                                                                                                            
print("log_flux:", data.log_fluxes)                       
print("log_err:", data.log_errors)
print("log_err_old:", data.errors / (data.fluxes * jnp.log(10)))

nuFnu_model = observed_sed(data.frequencies, p, cosmo)
log_model = jnp.log10(jnp.maximum(nuFnu_model, 1e-300))
residuals = (data.log_fluxes - log_model) / data.log_errors
old_ll = -0.5 * jnp.sum(residuals ** 2)

new_ll = log_likelihood(p, data=data, cosmo=cosmo)
print(f"old: {old_ll}, new: {new_ll}, diff: {old_ll - new_ll}")

