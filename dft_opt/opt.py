import jax 
import optax
import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as jnl
import optimistix as optx
from .orthonormalize import cayley
from pyscfad.gto import Mole
from pyscfad.dft import RKS
from pyscfad.scf import RHF


class Hamiltonian(eqx.Module):
    mol: Mole
    kernel: RHF
    ortho: str

    def __init__(self, mol, kernel, ortho_fn='qr'):
        self.mol = mol
        self.kernel = kernel
        self.ortho = ortho_fn
    
    def density_matrix(self, C):
        return jnp.einsum("k,ik,jk->ij", self.occupancy, C, C)

    def orthonormalize(self, Z):
        if self.ortho == "qr":
            return self.X @ jnl.qr(Z).Q
        elif self.ortho == "cayley":    
            return self.X @ cayley(Z)
        
    def __call__(self, P):
        vff = self.kernel.get_veff(P)
        return vff
    
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
        # Assumes uncharged systems in restricted Kohn-Sham
        occ = jnp.full(self.mol.nao, 2.0)
        mask = occ.cumsum() > self.mol.tot_electrons
        occ = jnp.where(mask, 0.0, occ)
        return occ
    
    @property
    def hcore(self):
        core = self.kernel.get_hcore(self.mol)
        return core


@jax.jit
@jax.value_and_grad
def energy(Z, H):
    C = H.X @ H.orthonormalize(Z)
    P = H.density_matrix(C)
    return H(P)


def solve(H, iter, _):
    n, k = H.X.shape
    Z_init = jax.device_put(jnp.eye(n), device=jax.devices()[0])
    # Z_init = jax.random.normal(jax.random.PRNGKey(42), (n, k)) / jnp.sqrt(n)

    solver = optx.BFGS(atol=1e-6, rtol=1e-5)
    aux = lambda Z, _: energy(Z, H)[0]
    sol = optx.minimise(aux, solver, Z_init, max_steps=4000)

    return sol.value, aux(sol.value, None), None


def optimize(H, iter, lr):
    n, k = H.X.shape
    Z = jax.device_put(jnp.eye(n), device=jax.devices()[0])
    # Z = jax.random.normal(jax.random.PRNGKey(0), (n, k)) / jnp.sqrt(n)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(Z)

    history = []
    for i in range(iter):
        e, grads = energy(Z, H)
        updates, opt_state = optimizer.update(grads, opt_state)
        Z = optax.apply_updates(Z, updates)
        history.append(e.item())

    return Z, e, history


def validate(Z, H, nelec):
    Q = H.orthonormalize(Z) 
    C = H.X @ Q
    P = H.density_matrix(C)
    S = jnp.linalg.inv(H.X @ H.X.T)

    assert jnp.allclose(jnp.eye(Z.shape[0]), Q.T @ Q, atol=1e-5), "Q is not orthonormal"
    assert jnp.allclose(nelec, jnp.trace(P @ S), atol=1e-5), f"Trace(P @ S) != N"

    # Hessian = jax.hessian(lambda X: energy(X, H)[0])(Z)
    # assert jnl.eigh(Hessian)[0].all() >= 0, "Hession != PSD"

