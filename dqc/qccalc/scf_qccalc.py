from __future__ import annotations
from abc import abstractmethod, abstractproperty
from dqc.system.base_system import BaseSystem
from dqc.qccalc.base_qccalc import BaseQCCalc
from dqc.utils.datastruct import SpinParam
from dft_opt import Cayley, QR
from typing import Optional, Dict, Any, List, Union, Tuple
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam
from dft_opt.bfgs import BFGS
from dft_opt.rbfgs import RBFGS
from tqdm import tqdm
import torch


def matexp(Z):
    A = torch.tril(Z, -1)
    X = A - A.T
    Q = torch.matrix_exp(X)
    return Q


def polar(m):
    U, _, Vh = torch.linalg.svd (m)
    u = U @ Vh
    # p = Vh.T.conj() @ S.diag().to (dtype = m.dtype) @ Vh
    return  u


def qr(Z):
    Q, _ = torch.linalg.qr(Z)
    return Q


def cayley(Z):
    A = torch.tril(Z, -1)
    X = A - A.T
    I = torch.eye(X.shape[0])
    Q = torch.linalg.solve(I + X, I - X)
    Q = Q @ Q
    return Q


def get_X(S):
    S = S.double()
    N = 1 / torch.sqrt(torch.diag(S))
    overlap = N.unsqueeze(1) * N.unsqueeze(0) * S
    s, U = torch.linalg.eigh(overlap)
    s = torch.diag(torch.pow(s, -0.5))
    X = U @ s @ U.T
    return X


def occupancy(occupancy, n):
    occ = torch.zeros(n)
    k = occupancy.shape[0]
    occ[:k] = occupancy
    return occ


def density_matrix(C, occupancy):
    return torch.einsum("k,ik,jk->ij", occupancy, C, C)


class SCF_QCCalc(BaseQCCalc):
    """
    Performing Restricted or Unrestricted self-consistent field iteration
    (e.g. Hartree-Fock or Density Functional Theory)

    Arguments
    ---------
    engine: BaseSCFEngine
        The SCF engine
    variational: bool
        If True, then use optimization of the free orbital parameters to find
        the minimum energy.
        Otherwise, use self-consistent iterations.
    """

    def __init__(self, engine: BaseSCFEngine):
        self._engine = engine
        self._polarized = engine.polarized
        self._shape = self._engine.shape
        self.dtype = self._engine.dtype
        self.device = self._engine.device
        self._has_run = False

    def get_system(self) -> BaseSystem:
        return self._engine.get_system()

    def run(self, opt_type, iter, lr) -> BaseQCCalc:

        if self._engine._hamilton._aoparamzer == "qr":
            ortho_fn = qr
        elif self._engine._hamilton._aoparamzer == "cayley":
            ortho_fn = cayley
        elif self._engine._hamilton._aoparamzer == "polar":
            ortho_fn = polar
        elif self._engine._hamilton._aoparamzer == "matexp":
            ortho_fn = matexp
        else:
            raise ValueError("Unknown orthogonalization function: %s" % self._engine._aoparamzer)

        n = self._engine._hamilton.get_kinnucl().shape[0]
        X = get_X(self._engine._hamilton._ovlp)
        orb_weights = self._engine._system.get_orbweight(polarized=self._polarized)

        if isinstance(orb_weights, SpinParam):
            occ = torch.zeros(2, n)

            occ_u = occupancy(orb_weights.u, n)
            occ_d = occupancy(orb_weights.d, n)

            occ[0, :len(occ_u)] = occ_u
            occ[1, :len(occ_d)] = occ_d

            init = torch.randn(2, n, n, dtype=torch.double)
            nelec = orb_weights.u.sum() + orb_weights.d.sum()
        else:
            occ = occupancy(orb_weights, n)
            init = torch.randn(n, n, dtype=torch.double)
            nelec = orb_weights.sum()

        if opt_type == "adam" or opt_type == "bfgs":
            Z = torch.nn.Parameter(init)
            optimizer = torch.optim.LBFGS([Z], line_search_fn="strong_wolfe", max_iter=1) if opt_type == "bfgs" else torch.optim.Adam([Z], lr=lr) 
            # optimizer = BFGS([Z], lr=lr) if opt_type == "bfgs" else torch.optim.Adam([Z], lr=lr) 
        elif opt_type == "rbfgs" or opt_type == "radam":
            assert self._engine._hamilton._aoparamzer in ["cayley", "qr"], "Riemannian BFGS only works with Cayley"
            man = Cayley() if self._engine._hamilton._aoparamzer == "cayley" else QR()
            man.set_S(X)

            if init.dim() == 3:
                init[0, :, :] = man.projx(init[0, :, :])
                init[1, :, :] = man.projx(init[1, :, :])
            else:
                init = man.projx(init)

            Z = ManifoldParameter(init, manifold=man)
            optimizer = RBFGS([Z], lr=lr) if opt_type == "rbfgs" else RiemannianAdam([Z], lr=lr)
        else:
            raise RuntimeError("Unknown optimizer: %s" % opt_type)
            
        out_dm, history = self.optimize(Z, X, occ, ortho_fn, optimizer, iter)

        self._dm = out_dm
        self._X = X
        self._Z = Z
        self._occupancy = occ
        self._nelec = nelec
        self.history = history
        self._has_run = True
        return self
    
    def optimize(self, Z, X, occ, ortho_fn, optimizer, iters) -> BaseQCCalc:
        
        def closure():
            optimizer.zero_grad()

            if Z.dim() == 3:

                occ1 = occ[0, :].squeeze()
                occ2 = occ[1, :].squeeze()

                Z1 = Z[0, :, :].squeeze()
                Z2 = Z[1, :, :].squeeze()

                C1 = X @ ortho_fn(Z1)
                C2 = X @ ortho_fn(Z2)

                P1 = density_matrix(C1, occ1)
                P2 = density_matrix(C2, occ2)
                P = P1 + P2

            else:
                
                C = X @ ortho_fn(Z)
                P = density_matrix(C, occ)

            loss = self._engine.dm2energy(P)
            loss.backward()
            return loss
        
        history = torch.zeros(iters)
        for i in (pbar := tqdm(range(iters))):

            if isinstance(optimizer, RiemannianAdam) or isinstance(optimizer, RBFGS):

                if Z.dim() == 3:
                    Z.data[0, :, :] = Z.manifold.projx(Z.data[0, :, :])
                    Z.data[1, :, :] = Z.manifold.projx(Z.data[1, :, :])
                else:
                    Z.data = Z.manifold.projx(Z.data)
            
            E = optimizer.step(closure)
            history[i] = E.item()
            pbar.set_description(f"Iter [{i}/{iters}] - Energy: {E.item():.6f}")

        if Z.dim() == 3:
            
            if isinstance(optimizer, RiemannianAdam):
                Z.data[0, :, :] = Z.manifold.projx(Z.data[0, :, :])
                Z.data[1, :, :] = Z.manifold.projx(Z.data[1, :, :])

            dm1 = density_matrix(X @ ortho_fn(Z[0, :, :]), occ[0, :].squeeze())
            dm2 = density_matrix(X @ ortho_fn(Z[1, :, :]), occ[1, :].squeeze())
            dm = dm1 + dm2

        else:

            if isinstance(optimizer, RiemannianAdam):
                Z.data = Z.manifold.projx(Z.data)

            dm = density_matrix(X @ ortho_fn(Z), occ)

        return dm, history

    def energy(self) -> torch.Tensor:
        # returns the total energy of the system
        assert self._has_run
        return self._engine.dm2energy(self._dm)

    def aodm(self) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        # returns the density matrix in the atomic-orbital basis
        assert self._has_run
        return self._dm

    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]):
        # calculate the energy given the density matrix
        assert (isinstance(dm, torch.Tensor) and not self._polarized) or \
            (isinstance(dm, SpinParam) and self._polarized)
        return self._engine.dm2energy(dm)

    def _get_zero_dm(self) -> Union[SpinParam[torch.Tensor], torch.Tensor]:
        # get the initial dm that are all zeros
        if not self._polarized:
            return torch.zeros(self._shape, dtype=self.dtype,
                               device=self.device)
        else:
            dm0_u = torch.zeros(self._shape, dtype=self.dtype,
                                device=self.device)
            dm0_d = torch.zeros(self._shape, dtype=self.dtype,
                                device=self.device)
            return SpinParam(u=dm0_u, d=dm0_d)

class BaseSCFEngine(torch.nn.Module):
    @abstractproperty
    def polarized(self) -> bool:
        """
        Returns if the system is polarized or not
        """
        pass

    @abstractproperty
    def shape(self):
        """
        Returns the shape of the density matrix in this engine.
        """
        pass

    @abstractproperty
    def dtype(self) -> torch.dtype:
        """
        Returns the dtype of the tensors in this engine.
        """
        pass

    @abstractproperty
    def device(self) -> torch.device:
        """
        Returns the device of the tensors in this engine.
        """
        pass

    @abstractmethod
    def get_system(self) -> BaseSystem:
        """
        Returns the system involved in the engine.
        """
        pass

    @abstractmethod
    def dm2energy(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Calculate the energy from the given density matrix.
        """
        pass

    @abstractmethod
    def dm2scp(self, dm: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Convert the density matrix into the self-consistent parameter (scp).
        Self-consistent parameter is defined as the parameter that is put into
        the equilibrium function, i.e. y in `y = f(y, x)`.
        """
        pass

    @abstractmethod
    def scp2dm(self, scp: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Calculate the density matrix from the given self-consistent parameter (scp).
        """
        pass

    @abstractmethod
    def scp2scp(self, scp: torch.Tensor) -> torch.Tensor:
        """
        Calculate the next self-consistent parameter (scp) for the next iteration
        from the previous scp.
        """
        pass

    @abstractmethod
    def aoparams2ene(self, aoparams: torch.Tensor, aocoeffs: torch.Tensor,
                     with_penalty: Optional[float] = None) -> torch.Tensor:
        """
        Calculate the energy from the given atomic orbital parameters and coefficients.
        """
        pass

    @abstractmethod
    def aoparams2dm(self, aoparams: torch.Tensor, aocoeffs: torch.Tensor,
                    with_penalty: Optional[float] = None) -> \
            Tuple[Union[torch.Tensor, SpinParam[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Calculate the density matrix and the penalty from the given atomic
        orbital parameters and coefficients.
        """
        pass

    @abstractmethod
    def pack_aoparams(self, aoparams: Union[torch.Tensor, SpinParam[torch.Tensor]]) -> torch.Tensor:
        """
        Pack the ao params into a single tensor.
        """
        pass

    @abstractmethod
    def unpack_aoparams(self, aoparams: torch.Tensor) -> Union[torch.Tensor, SpinParam[torch.Tensor]]:
        """
        Unpack the ao params into a tensor or SpinParam of tensor.
        """
        pass

    @abstractmethod
    def set_eigen_options(self, eigen_options: Dict[str, Any]) -> None:
        """
        Set the options for the diagonalization (i.e. eigendecomposition).
        """
        pass

    @abstractmethod
    def getparamnames(self, methodname: str, prefix: str = "") -> List[str]:
        """
        List all the names of parameters used in the given method.
        """
        pass
