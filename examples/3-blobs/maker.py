import jax.numpy as jnp
from model import simplemodel2
from relax import hist_kde


def yield_maker(
    data_gen,
    bandwidth=None,
    bins=jnp.array([-jnp.inf, *jnp.linspace(-100, 100, 3), jnp.inf]),
    SCALE=1,
    syst=True,
    reflect_infinities=True,
):
    def model(angle, anchr=jnp.array([0.0, 0.0])):

        s1, b1, b2, b3 = data_gen()

        # direc = jnp.array([jnp.cos(angle),jnp.sin(angle)])
        normal = jnp.array([jnp.cos(angle + jnp.pi / 2), jnp.sin(angle + jnp.pi / 2)])

        hb1 = hist_kde(
            jnp.matmul((b1 - anchr), normal),
            bins=bins,
            bandwidth=bandwidth,
            reflect_infinities=reflect_infinities,
        )
        hb2 = hist_kde(
            jnp.matmul((b2 - anchr), normal),
            bins=bins,
            bandwidth=bandwidth,
            reflect_infinities=reflect_infinities,
        )
        hb3 = hist_kde(
            jnp.matmul((b3 - anchr), normal),
            bins=bins,
            bandwidth=bandwidth,
            reflect_infinities=reflect_infinities,
        )

        hs1 = hist_kde(
            jnp.matmul((s1 - anchr), normal),
            bins=bins,
            bandwidth=bandwidth,
            reflect_infinities=reflect_infinities,
        )

        nb1 = (hb1) / SCALE
        nb2 = (hb2) / SCALE
        nb3 = (hb3) / SCALE
        ns1 = (hs1) / SCALE

        if not syst:
            nb1, nb2, nb3 = nb2, nb2, nb2

        m, bonlypars = simplemodel2(ns1, nb1, nb2, nb3)
        return m, bonlypars

    return model
