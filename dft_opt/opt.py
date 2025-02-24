import jax
import time
import optax
import equinox as eqx
import jax.numpy as jnp
import optimistix as optx


def get_solver(optimizer, atol=1e-4, rtol=1e-4, lr=0.001):
    if optimizer == "BFGS":
        return optx.BFGS(atol=atol, rtol=rtol)
    elif optimizer == "Adam":
        optim = optax.adam(learning_rate=lr)
        return optx.OptaxMinimiser(optim, atol=atol, rtol=rtol)
    elif optimizer == "CG":
        return optx.NonlinearCG (atol=atol, rtol=rtol)
    else:
        raise ValueError("Invalid optimizer")


def solve(H, iter, lr, optimizer):
    n, _ = H.X.shape
    Z_init = jnp.eye(n, device=H.X.device)
    
    @jax.jit
    def energy(Z, _):
        C = H.X @ H.orthonormalize(Z)
        P = H.density_matrix(C)
        out = H(P)
        return out

    solver = get_solver(optimizer=optimizer, lr=lr)

    start_time = time.time()
    sol = optx.minimise(energy, solver, Z_init, max_steps=iter)
    elapsed_time = (time.time() - start_time) * 1000

    return sol.value, energy(sol.value, None), elapsed_time 


def solve_with_history(H, iter, lr, optimizer):
    y = jnp.eye(H.X.shape[0])

    @jax.jit
    def energy(Z, _):
        C = H.X @ H.orthonormalize(Z)
        P = H.density_matrix(C)
        e = H(P)
        aux = None
        return e, aux

    solver = get_solver(optimizer=optimizer, lr=lr)

    args = None
    f_struct = jax.ShapeDtypeStruct((), jnp.float64)
    options = dict()
    aux_struct = None
    tags = frozenset()

    step = eqx.filter_jit(
        eqx.Partial(solver.step, fn=energy, args=args, options=options, tags=tags)
    )
    terminate = eqx.filter_jit(
        eqx.Partial(solver.terminate, fn=energy, args=args, options=options, tags=tags)
    )

    state = solver.init(energy, y, args, options, f_struct, aux_struct, tags)
    done, result = terminate(y=y, state=state)
    
    history = [energy(y, args)[0].item()]
    start_time = time.time()
    while not done and len(history) < iter:
        y, state, aux = step(y=y, state=state)
        done, result = terminate(y=y, state=state)
        history.append(energy(y, args)[0].item())

    if result != optx.RESULTS.successful:
        print("Optimization failed.", flush=True)

    return history
