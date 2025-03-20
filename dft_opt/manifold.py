from geoopt.manifolds.base import Manifold
from geoopt.tensor import ManifoldTensor
from geoopt.utils import size2shape
from torch.linalg import solve, svd, inv
from typing import Tuple, Optional, Union
from torch import Tensor, eye, cat, arange, zeros, randn, allclose, einsum, tril


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
        self.L = inv(X.T)
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

    def retr(self, C: Tensor, Z: Tensor) -> Tensor:
        """
        Retraction based on the Cayley transform:
        
        Ret(Z, C1) = [I - ½ W(Z, C1)S]⁻¹ [C1 + 0.5 Z]
        """

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
        
        Transp_{C1 -> Z}(Y) = [I - ½ W(Z, C1)S]⁻¹ [I - ½ W(Z, C1)S]Y
        Ret(Z, C1) = X_min⁻¹ X_plus Y, where X_min = I - ½ W(Z, C1)S, X_plus = I + ½ W(Z, C1)S
        """
        I = eye(Z.shape[-2], device=Z.device, dtype=Z.dtype)
        W = self._compute_W(C, Z)
        WS = W @ self.S 
        lhs = I - 0.5 * WS
        rhs = (I + 0.5 * WS) @ Y
        qv = solve(lhs, rhs)
        return qv

    def retr_transp(self, C: Tensor, Z: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:

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
        lhs = inv(self.S) @ G
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

        arg = U.T @ (self.S @ V)
        return arg.sum([-1, -2], keepdim=keepdim)
    

    ####################################
    # Check projection and retraction
    ####################################
    def projx(self, X: Tensor) -> Tensor:
        Y = self.L @ X
        U, _, V = svd(Y, full_matrices=False)
        Q = einsum("...ik,...kj->...ij", U, V)
        C = inv(self.L).T @ Q

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
    

# TODO: Implement Grassmanian manifold with it's polar map