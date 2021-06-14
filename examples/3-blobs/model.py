"""
Here you will find the model definition -- a simple 1-channel model that defines:
- a signal contribution s with normalisation factor mu (param of interest)
- a nominal background contribution that has two variations, 'bup' and 'bdown'
    - these are interpolated between to define a nuisance parameter 'uncorr_bkguncrt'
"""
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pyhf

jax_backend = pyhf.tensor.jax_backend(precision="64b")
pyhf.set_backend(jax_backend)


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
def simplemodel2(s, b_up, b_nom, b_dn):
    spec = {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": s,
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "background",
                        "data": b_nom,
                        "modifiers": [
                            {
                                "name": "uncorr_bkguncrt",
                                "type": "histosys",
                                "data": {"hi_data": b_up, "lo_data": b_dn},
                            }
                        ],
                    },
                ],
            }
        ]
    }

    m = pyhf.Model(spec)
    nompars = m.config.suggested_init()
    bonlypars = jnp.asarray([x for x in nompars])
    bonlypars = jax.ops.index_update(bonlypars, m.config.poi_index, 0.0)
    return m, bonlypars
