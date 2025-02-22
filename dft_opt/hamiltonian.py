import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from .orthonormalize import cayley

from pyscfad.gto import Mole
from pyscfad.dft import RKS
from pyscfad.scf import RHF


class Hamiltonian(eqx.Module):
    mol: Mole
    kernel: RHF

    def __init__(self, mol, kernel):
        self.mol = mol
        self.kernel = kernel
    
    def density_matrix(self, C):
        return jnp.einsum("k,ik,jk->ij", self.occupancy, C, C)

    def orthonormalize(self, Z):
        Q = jnl.qr(Z).Q
        # Q = cayley(Z)
        return Q

    def __call__(self, P):
        h1e = self.kernel.get_hcore()
        vhf = self.kernel.get_veff(dm=P)
        e_tot = self.kernel.energy_tot(dm=P, h1e=h1e, vhf=vhf)
        return e_tot
    
    @property
    def X(self):
        S = jnp.array(self.mol.intor(f"int1e_ovlp"))
        N = 1 / jnp.sqrt(jnp.diagonal(S))
        overlap = N[:, jnp.newaxis] * N[jnp.newaxis, :] * S

        s, U = jnl.eigh(overlap)
        s = jnp.diag(jnp.power(s, -0.5))
        X = U @ s @ U.T
        return X

    @property
    def occupancy(self):
        occ = jnp.full(self.mol.nao, 2.0)
        mask = occ.cumsum() > self.mol.tot_electrons()
        occ = jnp.where(mask, 0.0, occ)
        return occ
    
    @property
    def hcore(self):
        core = self.kernel.get_hcore(self.mol)
        return core
