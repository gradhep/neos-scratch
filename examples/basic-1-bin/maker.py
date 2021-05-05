import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pyhf

jax_backend = pyhf.tensor.jax_backend(precision="64b")
pyhf.set_backend(jax_backend)


def model_maker():
    @patch.object(sys.modules["pyhf.tensor.common"], "default_backend", new=jax_backend)
    @patch.object(sys.modules["pyhf.pdf"], "default_backend", new=jax_backend)
    @patch("pyhf.default_backend", new=jax_backend)
    @patch.object(
        sys.modules["pyhf.interpolators.code0"], "default_backend", new=jax_backend
    )
    @patch.object(
        sys.modules["pyhf.interpolators.code1"], "default_backend", new=jax_backend
    )
    @patch.object(
        sys.modules["pyhf.interpolators.code2"], "default_backend", new=jax_backend
    )
    @patch.object(
        sys.modules["pyhf.interpolators.code4"], "default_backend", new=jax_backend
    )
    @patch.object(
        sys.modules["pyhf.interpolators.code4p"], "default_backend", new=jax_backend
    )
    @patch.object(
        sys.modules["pyhf.modifiers.shapefactor"], "default_backend", new=jax_backend
    )
    @patch.object(
        sys.modules["pyhf.modifiers.shapesys"], "default_backend", new=jax_backend
    )
    @patch.object(
        sys.modules["pyhf.modifiers.staterror"], "default_backend", new=jax_backend
    )
    @patch.object(sys.modules["pyhf.constraints"], "default_backend", new=jax_backend)
    def make_spec(yields):
        s, b, bup, bdown = yields

        spec = {
            "channels": [
                {
                    "name": "nn",
                    "samples": [
                        {
                            "name": "signal",
                            "data": s,
                            "modifiers": [
                                {"name": "mu", "type": "normfactor", "data": None}
                            ],
                        },
                        {
                            "name": "bkg",
                            "data": b,
                            "modifiers": [
                                {
                                    "name": "nn_histosys",
                                    "type": "histosys",
                                    "data": {
                                        "lo_data": bdown,
                                        "hi_data": bup,
                                    },
                                }
                            ],
                        },
                    ],
                },
            ],
        }

        m = pyhf.Model(spec)
        nompars = m.config.suggested_init()
        bonlypars = jnp.asarray([x for x in nompars])
        bonlypars = jax.ops.index_update(bonlypars, m.config.poi_index, 0.0)
        return m, bonlypars

    return make_spec
