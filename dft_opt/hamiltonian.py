import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from .orthonormalize import cayley, qr
from pyscfad.gto import Mole
# from pyscfad.dft import RKS
from pyscfad.scf import RHF
from typing import Union
    

class Hamiltonian(eqx.Module):
    _mol: Mole
    _kernel: RHF
    _orthos: Union[cayley, qr]

    def __init__(self, mol: Mole, kernel: RHF, orthos: str):
        self._mol = mol
        self._kernel = kernel
        self._orthos = cayley if orthos == "cayley" else qr
    
    def density_matrix(self, C: jnp.ndarray) ->jnp.ndarray:
        return jnp.einsum("k,ik,jk->ij", self.occupancy, C, C)

    def orthonormalize(self, Z: jnp.ndarray) -> jnp.ndarray:
        return self._orthos(Z)

    def __call__(self, P: jnp.ndarray) -> jnp.ndarray:
        h1e = self._kernel.get_hcore()
        vhf = self._kernel.get_veff(dm=P)
        e_tot = self._kernel.energy_tot(dm=P, h1e=h1e, vhf=vhf)
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
