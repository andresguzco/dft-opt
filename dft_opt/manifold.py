from geoopt.manifolds.base import Manifold
from geoopt.tensor import ManifoldTensor
from geoopt.utils import size2shape
from torch.linalg import solve, svd, pinv
from typing import Tuple, Optional, Union
from torch import Tensor, eye, cat, arange, zeros, randn, allclose, einsum, tril, zeros_like
import torch

def cayley(Z):
    X = tril(Z, -1) - tril(Z, -1).T
    I = eye(X.shape[0])
    Q = solve(I + X, I - X)
    Q = Q @ Q
    return Q


class Cayley(Manifold):
    __doc__ = """
    Manifold induced by the following matrix constraint:

    .. math::

        C^\top S C = I\\
        C, S \in \mathrm{R}^{n\times n}\\
    """
    name = "Cayley"
    reversible = True
    ndim = 2

    def __new__(cls):
        return super().__new__(cls)

    def set_S(self, X: Tensor):
        self.L = pinv(X.T)
        self.S = self.L @ self.L.T
        return self
    
    def _compute_P(self, C: Tensor) -> Tensor:
        """
        Compute the skew‐symmetric matrix P based on the formula:
        
        P = I - 0.5 CCᵀS
        """
        I = eye(C.shape[-2], device=C.device, dtype=C.dtype)
        return I - 0.5 * (C @ (C.T @ self.S))

    def _compute_W(self, C: Tensor, Z: Tensor) -> Tensor:
        """
        Compute the skew‐symmetric matrix W based on the formula:
        
        W(Z, C1) = (I - ½ C1 C1ᵀ S) Z C1ᵀ - C1 Zᵀ (I - ½ C1 C1ᵀ S)ᵀ
        W(Z, C1) = A - Aᵀ, where A = (I - ½ C1 C1ᵀ S) Z C1ᵀ

        where x represents C1 and u represents Z.
        """
        P = self._compute_P(C)
        A = P @ Z @ C.T
        return A - A.T
    ####################################
    # Projection and retraction
    ####################################
    def retr(self, C: Tensor, Z: Tensor) -> Tensor:
        """
        Retraction based on the Cayley transform:
        
        Ret(Z, C1) = [I - ½ W(Z, C1)S]⁻¹ [C1 + 0.5 Z]
        """

        if C.dim() == 3:

            C1 = C[0, :, :].squeeze()
            C2 = C[1, :, :].squeeze()

            Z1 = Z[0, :, :].squeeze()
            Z2 = Z[1, :, :].squeeze()

            out = zeros_like(C)
            out[0, :, :] = self.retr(C1, Z1)
            out[1, :, :] = self.retr(C2, Z2)

        else:

            I = eye(Z.shape[-2], device=Z.device, dtype=Z.dtype)
            WS  = self._compute_W(C, Z) @ self.S
            lhs = I - 0.5 * WS
            rhs = C + 0.5 * Z
            out = solve(lhs, rhs)

        return out
    
    expmap = retr

    def transp_follow_retr(self, C: Tensor, Z: Tensor, Y: Tensor) -> Tensor:
        """
        Joint transport map from C1 to Z based of vector Y and point q:
        
        Transp_{C1 -> Z}(Y) = [I - ½ W(Z, C1)S]⁻¹ [I - ½ W(Z, C1)S]Y
        Ret(Z, C1) = X_min⁻¹ X_plus Y, where X_min = I - ½ W(Z, C1)S, X_plus = I + ½ W(Z, C1)S
        """
        I = eye(Z.shape[-2], device=Z.device, dtype=Z.dtype)
        W = self._compute_W(C, Z)
        WS = W @ self.S 
        lhs = I - 0.5 * WS
        rhs = (I + 0.5 * WS) @ Y
        lhs[..., arange(W.shape[-2]), arange(C.shape[-2])] += 1
        qY = solve(lhs, rhs)
        return qY

    transp_follow_expmap = transp_follow_retr

    def transp(self, C: Tensor, Z: Tensor, Y: Tensor) -> Tensor:
        """
        Transport map from C1 to Z based of vector Y:
        
        Transp_{C1 -> Z}(Y) = [I - ½ W(Z, C1)S]⁻¹ [I + ½ W(Z, C1)S]Y
        Ret(Z, C1) = X_min⁻¹ X_plus Y, where X_min = I - ½ W(Z, C1)S, X_plus = I + ½ W(Z, C1)S
        """

        if C.dim() == 3:
            C1 = C[0, :, :].squeeze()
            C2 = C[1, :, :].squeeze()

            Z1 = Z[0, :, :].squeeze()
            Z2 = Z[1, :, :].squeeze()

            Y1 = Y[0, :, :].squeeze()
            Y2 = Y[1, :, :].squeeze()

            qv = zeros_like(C)
            qv[0, :, :] = self.transp(C1, Z1, Y1)
            qv[1, :, :] = self.transp(C2, Z2, Y2)

        else:

            I = eye(Z.shape[-2], device=Z.device, dtype=Z.dtype)
            W = self._compute_W(C, Z)
            WS = W @ self.S 
            lhs = I - 0.5 * WS
            rhs = (I + 0.5 * WS) @ Y
            qv = solve(lhs, rhs)

        return qv

    def retr_transp(self, C: Tensor, Z: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:

        if C.dim() == 3:
            


            C1 = C[0, :, :].squeeze()
            C2 = C[1, :, :].squeeze()

            Z1 = Z[0, :, :].squeeze()
            Z2 = Z[1, :, :].squeeze()

            Y1 = Y[0, :, :].squeeze()
            Y2 = Y[1, :, :].squeeze()

            (new_x1, new_v1) = self.retr_transp(C1, Z1, Y1)
            (new_x2, new_v2) = self.retr_transp(C2, Z2, Y2)

            new_x = zeros_like(C)
            new_x[0, :, :] = new_x1
            new_x[1, :, :] = new_x2

            new_v = zeros_like(C)
            new_v[0, :, :] = new_v1
            new_v[1, :, :] = new_v2

        else:

            assert self._check_point_on_manifold(C)[0] is True, "C is not on the manifold"
            assert self._check_vector_on_tangent(C, Y)[0] is True, "Y is not on the tangent space"  
            assert self._check_vector_on_tangent(C, Z)[0] is True, "Z is not on the tangent space"

            C1Y = cat((C, Y), -1)
            qxvs = self.transp_follow_retr(C, Z, C1Y).view(
                C.shape[:-1] + (2, C.shape[-1])
            )
            new_x, new_v = qxvs.unbind(-2)

        return new_x, new_v

    expmap_transp = retr_transp

    def proju(self, C: Tensor, G: Tensor) -> Tensor:
        """
        Project the gradient G onto the tangent space of the Stiefel manifold at C1:
            G* = S^{-1} G - C1 (0.5 (C1ᵀ G + Gᵀ C1))
        """
        if G.dim() == 3:
            G_bar = zeros_like(G)
            G_bar[0, :, :] = self.proju(C[0, :, :].squeeze(0), G[0, :, :].squeeze(0))
            G_bar[1, :, :] = self.proju(C[1, :, :].squeeze(0), G[1, :, :].squeeze(0))
        
        else:

            lhs = pinv(self.S) @ G
            C1TG = C.T @ G
            GTC1 = G.T @ C
            rhs = C @ (0.5 * (C1TG + GTC1))
            G_bar = lhs - rhs

            assert self._check_vector_on_tangent(C, G_bar)[0] is True, "G_bar is not on the tangent space"
        return G_bar

    egrad2rgrad = proju
    
    def inner(self, x: Tensor, U: Tensor, V: Tensor = None, *, keepdim=False) -> Tensor:
        """
        Inner product for the Stiefel manifold with metric:
            <u, v>_x = tr(u^T S v)
        """
        if V is None:
            V = U

        if U.dim() == 3:
            out = zeros_like(U)
            out[0, :, :] = self.inner(x, U[0, :, :].squeeze(), V[0, :, :].squeeze())
            out[1, :, :] = self.inner(x, U[1, :, :].squeeze(), V[1, :, :].squeeze())

        else:
            arg = U.T @ (self.S @ V)
            out = arg.sum([-1, -2], keepdim=keepdim)

        return out 

    def projx(self, X: Tensor) -> Tensor:

        if X.dim() == 3:
            C = zeros_like(X)
            C[0, :, :] = self.projx(X[0, :, :].squeeze(0))
            C[1, :, :] = self.projx(X[1, :, :].squeeze(0))
        else:
            Y = self.L @ X
            U, _, V = svd(Y, full_matrices=False)
            Q = einsum("...ik,...kj->...ij", U, V)
            C = pinv(self.L).T @ Q

        assert self._check_point_on_manifold(C)[0] is True, "C was not projected to the manifold"

        return C
    
    ####################################
    # Checks for the Stiefel manifold
    ####################################
    def _check_shape(

        self, shape: Tuple[int], name: str
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok, reason = super()._check_shape(shape, name)
        if not ok:
            return False, reason
        shape_is_ok = shape[-1] <= shape[-2]
        if not shape_is_ok:
            return (
                False,
                f"`{name}` should have shape[-1] <= shape[-2], got {shape[-1]} </= {shape[-2]}"
            )
        return True, None

    def _check_point_on_manifold(
        self, C: Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ctsc = C.T @ (self.S @ C)
        # less memory usage for substract diagonal
        ctsc[..., arange(C.shape[-1]), arange(C.shape[-1])] -= 1
        ok = allclose(ctsc, ctsc.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, f"`C^T S C != I` with atol={atol}, rtol={rtol}"
        return True, None

    def _check_vector_on_tangent(
        self, C:Tensor, Z: Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        A = C.T @ (self.S @ Z)
        diff = A + A.T
        ok = allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, f"`C1^T S Z + Z^T S C1 !=0` with atol={atol}, rtol={rtol}"
        return True, None

    ####################################
    # Methods to sample and get origin
    ####################################

    def random_naive(self, *size, dtype=None, device=None) -> Tensor:
        self._assert_check_shape(size2shape(*size), "x")
        tens = randn(*size, device=device, dtype=dtype)
        return ManifoldTensor(cayley(tens), manifold=self)

    random = random_naive

    def origin(self, *size, dtype=None, device=None, seed=42) -> Tensor:
        self._assert_check_shape(size2shape(*size), "x")
        eye = zeros(*size, dtype=dtype, device=device)
        eye[..., arange(eye.shape[-1]), arange(eye.shape[-1])] += 1
        return ManifoldTensor(eye, manifold=self)
    


class QR(Manifold):
    __doc__ = """

    Manifold induced by the following matrix constraint:

    .. math::

        C^\top S C = I\\
        C, S \in \mathrm{R}^{n\times n}\\
    """
    name = "QR"
    reversible = True
    ndim = 2

    def __new__(cls):
        return super().__new__(cls)

    def set_S(self, X: Tensor):
        self.L = pinv(X.T)
        self.S = self.L @ self.L.T
        return self
    
    ####################################
    # Checks for the Stiefel manifold
    ####################################
    def _check_shape(
        self, shape: Tuple[int], name: str
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok, reason = super()._check_shape(shape, name)
        if not ok:
            return False, reason
        shape_is_ok = shape[-1] <= shape[-2]
        if not shape_is_ok:
            return (
                False,
                "`{}` should have shape[-1] <= shape[-2], got {} </= {}".format(
                    name, shape[-1], shape[-2]
                ),
            )
        return True, None

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        xtx = x.transpose(-1, -2) @ x
        # less memory usage for substract diagonal
        xtx[..., torch.arange(x.shape[-1]), torch.arange(x.shape[-1])] -= 1
        ok = torch.allclose(xtx, xtx.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`X^T X != I` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        diff = u.transpose(-1, -2) @ x + x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u^T x + x^T u !=0` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    ####################################
    # Methods to sample and get origin
    ####################################
    def random_naive(self, *size, dtype=None, device=None) -> torch.Tensor:
        self._assert_check_shape(size2shape(*size), "x")
        tens = torch.randn(*size, device=device, dtype=dtype)
        return ManifoldTensor(linalg.qr(tens)[0], manifold=self)

    random = random_naive

    def origin(self, *size, dtype=None, device=None, seed=42) -> torch.Tensor:
        self._assert_check_shape(size2shape(*size), "x")
        eye = torch.zeros(*size, dtype=dtype, device=device)
        eye[..., torch.arange(eye.shape[-1]), torch.arange(eye.shape[-1])] += 1
        return ManifoldTensor(eye, manifold=self)

    ####################################
    # Method to project and retract
    ####################################

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        if X.dim() == 3:
            C = zeros_like(X)
            C[0, :, :] = self.projx(X[0, :, :].squeeze(0))
            C[1, :, :] = self.projx(X[1, :, :].squeeze(0))
        else:
            U, _, V = linalg.svd(x, full_matrices=False)
            C = torch.einsum("...ik,...kj->...ij", U, V)

        return C

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            G_bar = torch.zeros_like(x)
            G_bar[0, :, :] = self.proju(x[0, :, :].squeeze(), u[0, :, :].squeeze())
            G_bar[1, :, :] = self.proju(x[1, :, :].squeeze(), u[1, :, :].squeeze())

        else:
             G_bar = u - x @ linalg.sym(x.transpose(-1, -2) @ u)
 
        return G_bar

    egrad2rgrad = proju

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            out = torch.zeros_like(x)
            out[0, :, :] = self.transp(x[0, :, :].squeeze(), y[0, :, :].squeeze(), v[0, :, :].squeeze())
            out[1, :, :] = self.transp(x[1, :, :].squeeze(), y[1, :, :].squeeze(), v[1, :, :].squeeze())
 
        else:
            out = self.proju(y, v)

        return out

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            v = u

        if u.dim() == 3:
            out = zeros_like(u)
            out[0, :, :] = self.inner(x, u[0, :, :].squeeze(), v[0, :, :].squeeze())
            out[1, :, :] = self.inner(x, u[1, :, :].squeeze(), v[1, :, :].squeeze())

        else:
            out = (u * v).sum([-1, -2], keepdim=keepdim)

        return out

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            q = zeros_like(x)
            q[0, :, :] = self.retr(x[0, :, :].squeeze(), u[0, :, :].squeeze())
            q[1, :, :] = self.retr(x[1, :, :].squeeze(), u[1, :, :].squeeze())
        else:
            q, r = linalg.qr(x + u)
            unflip = linalg.extract_diag(r).sign().add(0.5).sign()
            q *= unflip[..., None, :]
        return q

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            y = zeros_like(x)
            y[0, :, :] = self.expmap(x[0, :, :].squeeze(), u[0, :, :].squeeze())
            y[1, :, :] = self.expmap(x[1, :, :].squeeze(), u[1, :, :].squeeze())
        else:
            xtu = x.transpose(-1, -2) @ u
            utu = u.transpose(-1, -2) @ u
            eye = torch.zeros_like(utu)
            eye[..., torch.arange(utu.shape[-2]), torch.arange(utu.shape[-2])] += 1
            logw = linalg.block_matrix(((xtu, -utu), (eye, xtu)))
            w = linalg.expm(logw)
            z = torch.cat((linalg.expm(-xtu), torch.zeros_like(utu)), dim=-2)
            y = torch.cat((x, u), dim=-1) @ w @ z
        return y


# TODO: Implement Grassmanian manifold with it's polar map
