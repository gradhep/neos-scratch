# neos-scratch
various studies and demos relating to neos -- see gradhep/neos for the full repo


### note -- this is a python 3.7/3.8 compatible project only!

## dev

to keep formatting nice, initialize pre-commit (after possible `python -m pip install pre-commit`):

```pre-commit install```

install dependencies in venv of choice. note: `fax` breaks without an old enough `jax` installation, as it uses `jax.custom_transforms` which is now deprecated, so i pinned the `jax` and `jaxlib` versions.

```python -m pip install -r requirements.txt```
