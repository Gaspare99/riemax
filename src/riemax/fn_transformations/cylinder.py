import jax
import jax.numpy as jnp


def cylinder(x: jax.Array) -> jax.Array:
    """Cylinder global parametrization.
    cylinder: [0, 2Pi) x R --> R^3

    (theta, t) |--> (rho * cos(theta), rho * sin(theta), t)


    Parameters:
        x: local coordinate vector. x[0]=rho, x[1], t
        rho: radius of the circle 

    Returns:
        cylinder embedded in R^3. (rho * cos(x[0]), rho * sin(x[0]), x[1])
    """
    rho = 1
    x_1=rho * jnp.cos(x[0])
    x_2=rho * jnp.sin(x[0])
    x_3=x[1]

    return jnp.array([x_1 , x_2, x_3])
