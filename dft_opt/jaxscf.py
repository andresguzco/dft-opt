import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from pyscfad.gto import Mole


HERMITIAN = 1
ANTIHERMI = 2


@jax.jit
def energy_tot(mf, dm=None, h1e=None, vhf=None, dispersion=None):
    vhf = get_veff(dm=dm)
    e1 = jnp.einsum('ij,ji->', h1e, dm).real
    e_coul = jnp.einsum('ij,ji->', vhf, dm).real * .5
    e_tot = e1 + e_coul + dispersion
    return e_tot

@jax.jit
def Hamiltonian(mol, P, h1e, dispersion):
    vhf = get_veff(mol, P)
    e_elec = energy_tot(P, h1e, vhf, dispersion)
    return e_elec


def get_veff(mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
    vj, vk = get_jk(mol, jnp.asarray(dm), hermi, vhfopt)
    return vj - vk * .5


def get_jk(mol, dm, hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None):
    dm = jnp.asarray(dm, order='C')
    dm_shape = dm.shape
    dm_dtype = dm.dtype
    nao = dm_shape[-1]

    if dm_dtype == jnp.complex128:
        dm = jnp.vstack((dm.real, dm.imag)).reshape(-1,nao,nao)
        hermi = 0

    with mol.with_range_coulomb(omega):
        vj, vk = direct(dm, mol._atm, mol._bas, mol._env, vhfopt, hermi, mol.cart, with_j, with_k)

    if dm_dtype == jnp.complex128:
        if with_j:
            vj = vj.reshape((2,) + dm_shape)
            vj = vj[0] + vj[1] * 1j
        if with_k:
            vk = vk.reshape((2,) + dm_shape)
            vk = vk[0] + vk[1] * 1j
    else:
        if with_j:
            vj = vj.reshape(dm_shape)
        if with_k:
            vk = vk.reshape(dm_shape)
    return vj, vstack

def direct(dms, atm, bas, env, vhfopt=None, hermi=0, cart=False,
           with_j=True, with_k=True, out=None, optimize_sr=False):
    dms = jnp.asarray(dms, order='C', dtype=jnp.double)
    dms_shape = dms.shape
    nao = dms_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    n_dm = dms.shape[0]

    if vhfopt is None:
        cvhfopt = None
        cintopt = None
        if cart:
            intor = 'int2e_cart'
        else:
            intor = 'int2e_sph'
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        intor = vhfopt._intor

    vj = vk = None
    jkscripts = []
    n_jk = 0
    if with_j:
        jkscripts.extend(['ji->s2kl']*n_dm)
        n_jk += 1
    if with_k:
        if hermi == 1:
            jkscripts.extend(['li->s2kj']*n_dm)
        else:
            jkscripts.extend(['li->s1kj']*n_dm)
        n_jk += 1
    if n_jk == 0:
        return vj, vk

    dms = list(dms) * n_jk
    if out is None:
        out = jnp.empty((n_jk*n_dm, nao, nao))
    nr_direct_drv(intor, 's8', jkscripts, dms, 1, atm, bas, env,
                  cvhfopt, cintopt, out=out, optimize_sr=optimize_sr)
    if with_j and with_k:
        vj = out[:n_dm]
        vk = out[n_dm:]
    elif with_j:
        vj = out
    else:
        vk = out

    if with_j:
        for i in range(n_dm):
            lib.hermi_triu(vj[i], 1, inplace=True)
        vj = vj.reshape(dms_shape)
    if with_k:
        if hermi != 0:
            for i in range(n_dm):
                lib.hermi_triu(vk[i], hermi, inplace=True)
        vk = vk.reshape(dms_shape)

    return vj, vk


def hermi_triu(mat, hermi=HERMITIAN, inplace=True):

    assert (hermi == HERMITIAN or hermi == ANTIHERMI)

    if not inplace:
        mat = mat.copy('A')

    if mat.flags.c_contiguous:
        buf = mat
    elif mat.flags.f_contiguous:
        buf = mat.T
    else:
        raise NotImplementedError

    nd = mat.shape[0]
    assert (mat.size == nd**2)

    if mat.dtype == jnp.double:
        fn = _np_helper.NPdsymm_triu
    elif mat.dtype == jnp.complex128:
        fn = _np_helper.NPzhermi_triu
    else:
        raise NotImplementedError

    fn.restype = ctypes.c_void_p
    fn(ctypes.c_int(nd), buf.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(hermi))
    return mat


def get_X(mol):
    S = jnp.array(mol.intor(f"int1e_ovlp"))
    N = 1 / jnp.sqrt(jnp.diagonal(S))
    overlap = N[:, jnp.newaxis] * N[jnp.newaxis, :] * S
    s, U = jnl.eigh(overlap)
    s = jnp.diag(jnp.power(s, -0.5))
    X = U @ s @ U.T
    return X


@jax.jit
def occupancy(mol) -> jnp.ndarray:
    occ = jnp.full(mol.nao, 2.0)
    mask = occ.cumsum() > mol.tot_electrons()
    occ = jnp.where(mask, 0.0, occ)
    return occ


@jax.jit
def density_matrix(C: jnp.ndarray, occupancy: jnp.ndarray) ->jnp.ndarray:
    return jnp.einsum("k,ik,jk->ij", occupancy, C, C)


