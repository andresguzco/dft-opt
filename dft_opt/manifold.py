from geoopt.manifolds.stiefel import Stiefel
from torch.linalg import solve, cholesky
from typing import Tuple
from torch import Tensor, eye, trace, cholesky_solve

class CayleyStiefel(Stiefel):
    reversible = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_S(self, S: Tensor):
        self.name = f"CayleyStiefel"
        self.S = S
        return self

    def _compute_W(self, C1: Tensor, Z: Tensor) -> Tensor:
        """
        Compute the skew‐symmetric matrix W based on the formula:
        
        W(Z, C1) = (I - ½ C1 C1ᵀ S) Z C1ᵀ - C1 Zᵀ (I - ½ C1 C1ᵀ S)
        W(Z, C1) = A - Aᵀ, where A = (I - ½ C1 C1ᵀ S) Z C1ᵀ

        where x represents C1 and u represents Z.
        """
        I = eye(C1.shape[-2], device=C1.device, dtype=C1.dtype)
        A = I - 0.5 * (C1 @ (C1.T @ (self.S @ (Z @ C1.T))))
        return A - A.T
    
    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u - x @ u.transpose(-1, -2) @ x

    egrad2rgrad = proju

    def retr(self, C1: Tensor, Z: Tensor) -> Tensor:
        """
        Retraction based on the Cayley transform:
        
        Ret(Z, C1) = [I - ½ W(Z, C1)S]⁻¹ [C1 + 0.5 Z]
        """

        I = eye(Z.shape[-2], device=Z.device, dtype=Z.dtype)
        W = self._compute_W(Z, C1)
        retr_X = solve(C1 + 0.5 * Z, I - 0.5 * (W @ self.S))
        return retr_X
    
    expmap = retr

    def transp(self, C1: Tensor, Z: Tensor, Y: Tensor) -> Tensor:
        """
        Transport map from C1 to Z based of vector Y:
        
        Transp_{C1 -> Z}(Y) = [I - ½ W(Z, C1)S]⁻¹ [I - ½ W(Z, C1)S]Y
        Ret(Z, C1) = X_min⁻¹ X_plus Y, where X_min = I - ½ W(Z, C1)S, X_plus = I + ½ W(Z, C1)S
        """
        I = eye(Z.shape[-2], device=Z.device, dtype=Z.dtype)

        W = self._compute_W(Z, C1)
        X_min = I - 0.5 * (W @ self.S)
        X_plus = I + 0.5 * (W @ self.S)

        transp_X = solve(X_plus @ Y, X_min)

        return transp_X

    def retr_transp(self, x: Tensor, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Jointly computes the retraction and vector transport:
        
        new_x = [I - ½ W(Z, C1)S]⁻¹ [C1 + ½ Z]
        new_v = [I - ½ W(Z, C1)S]⁻¹ [I + ½ W(Z, C1)S] v
        """
        I = eye(x.shape[-2], device=x.device, dtype=x.dtype)
        W = self._compute_W(x, u)
        new_x = solve(I - 0.5 * (W @ self.S), x + 0.5 * u)
        new_v = solve(I - 0.5 * (W @ self.S), (I + 0.5 * (W @ self.S)) @ v)
        return new_x, new_v

    expmap_transp = retr_transp

    def proju(self, C1: Tensor, G: Tensor) -> Tensor:
        lhs = cholesky_solve(G, cholesky(self.S))
        rhs = C1 @ (0.5 * (C1.T @ G + G.T @ C1))
        return lhs - rhs

    egrad2rgrad = proju

    def inner(self, x: Tensor, U: Tensor, V: Tensor = None, *, keepdim=False) -> Tensor:
            """
            Inner product for the Stiefel manifold with metric:
                <u, v>_x = tr(u^T S v)
            """
            if V is None:
                V = U
            arg = U.T @ self.S @ V
            return trace(arg)