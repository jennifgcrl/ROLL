# Setup Notes

This project depends on transformer-engine[pytorch] and flash-attn, which can't be built with build isolation.
So, it's necessary to install some packages first:

```
uv pip install setuptools psutil numpy torch
```

Then, sync or build as usual:

```
uv sync
```

```
uv build
```
