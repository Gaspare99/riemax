import typing as tp

import riemax as rx

import einops

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

jax.config.update('jax_platform_name', 'cpu')

fn_transformation = rx.fn_transformations.cylinder
manifold = rx.Manifold.from_fn_transformation(fn_transformation)

from riemax.manifold.types import M, TpM, Rn

import jax
import jax.numpy as jnp
import numpy as np


# defining a point p on the manifold
p: M[jax.Array] = jnp.array([0.0, 0.0])

# defining vectors at p
v: TpM[jax.Array] = jnp.array([1.0, 0.0])
w: TpM[jax.Array] = jnp.array([0.0, 1.0])

metric_at_p = manifold.metric_tensor(p)
christoffel_symbols_at_p = manifold.sk_christoffel(p)

print(metric_at_p)
print(christoffel_symbols_at_p)

@jax.jit
def compute_curvature(x: jax.Array) -> jax.Array:
    return jnp.abs(manifold.ricci_scalar(x))

import haiku as hk
import optax

from riemax.ml.architectures import MLP                # your backbone
from riemax.ml.eikonal.model import TwowayEikonal        # Eq.18 wrapper
from riemax.ml.eikonal.losses import construct_eikonal_loss  #  | ⟨∇φ,∇φ⟩₉ − 1
from riemax.ml.training import construct_initialiser, construct_update  # JAX loop utils


#no standardization
identity = lambda x: x

def forward(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    # >>> MODULE CREATION HAPPENS INSIDE forward <<<

    # a) Build your raw network: 3 hidden layers of 128 units, then 1-unit output
    mlp = MLP(output_sizes=[128, 128, 128, 1])

    # b) Wrap it in the TwowayEikonal constraint (this is also an hk.Module)
    phi_model = TwowayEikonal(
        module=mlp,
        fn_transformation=fn_transformation,
        fn_standardise=identity,
    )

    # c) Call it to produce your φ_θ(q; p)
    return phi_model(p, q)

# 2) Now transform that into pure init/apply fns
transformed = hk.transform(forward)


# # 3) Use those in your script
# rng = jax.random.PRNGKey(0)
# # assume p_batch, q_batch are arrays of shape (B, dim)
# N = 100
# # Scale two independent Uniform[0,1) columns by [2π,1]
# p_batch = np.random.rand(N, 2) * jnp.array([2*jnp.pi, 1.0])
# q_batch = np.random.rand(N, 2) * jnp.array([2*jnp.pi, 1.0])
# params = transformed.init(rng, p_batch, q_batch)       # This traces forward once
# phi_vals = transformed.apply(params, rng, p_batch, q_batch)  # and this runs it

# # 4) Plug into your loss / optimizer loop as usual...
# eik_loss_fn = construct_eikonal_loss(transformed.apply, fn_transformation)

rng = jax.random.PRNGKey(0)
phi_fn = lambda params, p, q: transformed.apply(params, rng, p, q)

# 2) Build the Eikonal residual loss L(θ; p, q) = ⟨∇φ,∇φ⟩_g − 1
eik_loss_fn = construct_eikonal_loss(phi_fn, manifold.metric_tensor)

# 3) Pick an optimizer and wrap init/update
opt       = optax.adam(1e-3)
init_fn   = construct_initialiser(transformed.init, opt)
update_fn = construct_update(eik_loss_fn,      opt)

# 4) Prepare a batch of (p, q) samples in your coordinates
N = 1024
p_batch = np.random.rand(1, 2) * np.array([2*np.pi, 1.0])
q_batch = np.random.rand(1, 2) * np.array([2*np.pi, 1.0])
p_batch=p_batch.T
q_batch=q_batch.T

# 5) Initialize parameters + Optax state
state = init_fn(rng, p_batch, q_batch)

# 6) Training loop: each step returns (loss, new_state)
num_steps = 50000
for i in range(num_steps):
    # Call update_fn with your current state + data
    (loss, aux), state = update_fn(state, p_batch, q_batch)

    if i % 1000 == 0:
        print(f"step={i:04d}, loss={loss:.6f}, aux={aux}")

# 7) Extract final params & distance function
params     = state.params
distance_fn = lambda p, q: transformed.apply(params, rng, p, q)
