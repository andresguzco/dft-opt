import jax
import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from pyscfad.scf import RHF
from pyscfad.gto import Mole


class Hamiltonian(eqx.Module):
    _mol: Mole
    _kernel: RHF

    def __init__(self, mol: Mole, kernel: RHF):
        self._mol = mol
        self._kernel = kernel
    
    @jax.jit
    def density_matrix(self, C: jnp.ndarray) ->jnp.ndarray:
        return jnp.einsum("k,ik,jk->ij", self.occupancy, C, C)

    @jax.jit
    def __call__(self, P: jnp.ndarray) -> jnp.ndarray:
        vhf = self._kernel.get_veff(dm=P)
        e_tot = self._kernel.energy_tot(dm=P, h1e=self.hcore, vhf=vhf)
        return e_tot
    
    @property
    def X(self) -> jnp.ndarray:
        S = jnp.array(self._mol.intor(f"int1e_ovlp"))
        N = 1 / jnp.sqrt(jnp.diagonal(S))
        overlap = N[:, jnp.newaxis] * N[jnp.newaxis, :] * S

        s, U = jnl.eigh(overlap)
        s = jnp.diag(jnp.power(s, -0.5))
        X = U @ s @ U.T
        return X

    @property
    def occupancy(self) -> jnp.ndarray:
        occ = jnp.full(self._mol.nao, 2.0)
        mask = occ.cumsum() > self._mol.tot_electrons()
        occ = jnp.where(mask, 0.0, occ)
        return occ
    
    @property
    def hcore(self) -> jnp.ndarray:
        core = self._kernel.get_hcore(self._mol)
        return core
