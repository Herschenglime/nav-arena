# mmcv 2.1.0 for VIPlanner (aarch64, Python 3.12, torch 2.10+cu130, GB10 / SM 12.1)

`mmcv` must be compiled with CUDA ops; no matching wheel is published. Build it once from source and
install the resulting wheel into the Isaac Lab uv environment:

```bash
source setup.env
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=4 \
  uv pip wheel --no-deps --no-build-isolation --no-binary=mmcv mmcv==2.1.0 -w /tmp/mmcv_wheel
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps /tmp/mmcv_wheel/mmcv-2.1.0-*.whl
python -c "from mmcv import ops"   # verifies the compiled extension loads
```

Requires `nvcc` (`/usr/local/cuda`) and a C++ toolchain. A wheel built this way on another machine
with the same architecture, Python, torch, and CUDA versions can be copied and installed directly.
