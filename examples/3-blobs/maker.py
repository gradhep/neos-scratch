import jax.numpy as jnp
from model import simplemodel2
from relax import hist_kde


def yield_maker(
    data_gen, bandwidth=None, bins=jnp.linspace(-100, 100, 3), SCALE=1, syst=True
):
    def model(angle, anchr=jnp.array([0.0, 0.0])):

        s1, b1, b2, b3 = data_gen()

        # direc = jnp.array([jnp.cos(angle),jnp.sin(angle)])
        normal = jnp.array([jnp.cos(angle + jnp.pi / 2), jnp.sin(angle + jnp.pi / 2)])

        bins = jnp.linspace(-100, 100, 3)
        hb1 = hist_kde(jnp.matmul((b1 - anchr), normal), bins=bins, bandwidth=bandwidth)
        hb2 = hist_kde(jnp.matmul((b2 - anchr), normal), bins=bins, bandwidth=bandwidth)
        hb3 = hist_kde(jnp.matmul((b3 - anchr), normal), bins=bins, bandwidth=bandwidth)

        hs1 = hist_kde(jnp.matmul((s1 - anchr), normal), bins=bins, bandwidth=bandwidth)

        nb1 = hb1 / SCALE
        nb2 = hb2 / SCALE
        nb3 = hb3 / SCALE

        ns1 = hs1 / SCALE

        if not syst:
            nb1, nb2, nb3 = nb2, nb2, nb2

        m, bonlypars = simplemodel2(ns1, nb1, nb2, nb3)
        return m, bonlypars

    return model
