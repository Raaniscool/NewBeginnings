#!/usr/bin/env bash
# =============================================================================
# NewBeginnings environment setup (CPU-only, storage-lean).
#
# What this does, and why:
#   1. pip install the standard PyPI torch wheel with --no-deps (its declared
#      nvidia-* / triton / cuda-toolkit deps are ~5-6 GB of GPU libraries this
#      CPU-only project will never execute).
#   2. pip install the small real dependencies (numpy, pytest, tokenizers,
#      plus torch's python-level deps).
#   3. Generate stub .so files for the CUDA SONAMEs the torch libs link
#      against (scripts/setup_torch_cpu_stubs.py) so `import torch` works
#      without the nvidia stack.
#   4. Install a .pth preload that initializes system libstdc++ before torch's
#      closure loads (avoids a static-init-order iostream/locale crash seen on
#      this class of machine).
#   5. Verify: import torch, run a real CPU training micro-step.
#
# Everything here targets the machine's python environment (outside the repo);
# the repo itself stays clean and small. Re-running is safe (idempotent).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== NewBeginnings env setup =="

PYBIN="${PYBIN:-python3}"
SITE=$("$PYBIN" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"

if ! "$PYBIN" -m pip show torch >/dev/null 2>&1; then
  echo "-- installing torch ${TORCH_VERSION} (PyPI, --no-deps) ..."
  "$PYBIN" -m pip install --no-cache-dir --break-system-packages --no-deps \
      "torch==${TORCH_VERSION}"
fi

echo "-- installing python-level deps ..."
"$PYBIN" -m pip install --no-cache-dir --break-system-packages \
    "numpy>=1.26,<3" "pytest>=8,<10" "tokenizers>=0.15,<0.24" \
    filelock typing-extensions sympy networkx jinja2 fsspec

echo "-- generating CPU-only CUDA stubs ..."
"$PYBIN" scripts/setup_torch_cpu_stubs.py

echo "-- installing libstdc++ preload (.pth) ..."
cat > "$SITE/_torch_cpu_preload.py" <<'PYEOF'
"""NewBeginnings environment workaround - see scripts/setup_env.sh.
Pre-initializes system libstdc++ before torch's dlopen closure, avoiding a
static-init-order iostream/locale crash when `import torch` runs in a fresh
process on CPU-only machines where the CUDA stack is stubbed out."""
try:
    import ctypes, ctypes.util, os
    _name = ctypes.util.find_library("stdc++")
    if _name:
        ctypes.CDLL(_name,
                    mode=getattr(os, "RTLD_GLOBAL", 0x100) | getattr(os, "RTLD_NOW", 2))
except Exception:
    pass
PYEOF
echo "import _torch_cpu_preload" > "$SITE/_torch_cpu_preload.pth"

echo "-- verifying ..."
"$PYBIN" - <<'PYEOF'
import torch
from torch.nn import functional as F
torch.set_num_threads(2)
x = torch.randn(8, 8)
_ = F.scaled_dot_product_attention(x, x, x, is_causal=True)
lin = torch.nn.Linear(8, 8)
loss = lin(x).sum()
loss.backward()
opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
opt.step()
print("VERIFY_OK torch", torch.__version__)
PYEOF
echo "== env setup complete =="
