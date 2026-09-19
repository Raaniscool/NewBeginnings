#!/usr/bin/env bash
# Post-install: wipe NVIDIA/cu* deps pulled in by torch, then build minimal
# STUB shared libraries for every cuda NEEDED of torch's .so files.
# Stubs must satisfy the dynamic loader (symbol presence + version defs);
# they are NOT meant to be executed (torch CPU path never calls them).
#
# IMPORTANT: every function stub returns 0. Empty void bodies leave garbage
# in rax, which some callers (CUPTI registration & friends) dereference.
set -euo pipefail
T=/usr/local/lib/python3.11/dist-packages/torch
SP=/usr/local/lib/python3.11/dist-packages
TAG_RE='^[A-Za-z_]'

echo "== removing NVIDIA packages =="
mapfile -t PKGS < <(pip list --format=freeze | grep -Ei '^(nvidia-|triton|cupti|jiter)' | cut -d= -f1 || true)
if [ "${#PKGS[@]}" -gt 0 ]; then pip uninstall -y "${PKGS[@]}" || true; fi

echo "== building stub libs =="
D=$T/lib/stubs; rm -rf "$D"; mkdir -p "$D" "$D/src"

# soname -> reference .so to scrape symbols from (only used if still present)
declare -A REF=(
  [libcudnn.so.9]=libcudnn.so.9.19.0
  [libcufft.so.12]=libcufft.so.12.0.122
  [libcublas.so.13]=libcublas.so.13.2.0.322
  [libcublasLt.so.13]=libcublasLt.so.13.2.0.322
  [libcuda.so.1]=libcuda.so.610.30.1
  [libnccl.so.2]=libnccl.so.2.29.3
  [libcudart.so.13]=libcudart.so.13.2.49
  [libcupti.so.13]=libcupti.so.13.1.133
  [libcusparse.so.13]=libcusparse.so.13.0.3.45
  [libcusparseLt.so.0]=libcusparseLt.so.0.8.1.1
  [libcurand.so.10]=libcurand.so.10.4.1.408
  [libnvJitLink.so.13]=libnvJitLink.so.13.1.115
  [libnvtx3interop.so.1]=libnvtx3interop.so.1.0.0
  [libcufile.so.0]=libcufile_rdma.so.1.10.1
)
LDDIR=$SP/nvidia

for tgt in "${!REF[@]}"; do
  ref="${REF[$tgt]}"
  f=""
  [ -n "$ref" ] && f=$(find "$LDDIR" -name "$ref" 2>/dev/null | head -1 || true)
  src="$D/src/${tgt}.c"
  if [ -z "$f" ] && [ -s "$D/$tgt" ] && [ -s "$src" ]; then
    echo "  keep: $tgt (reference gone, existing stub preserved)"
    continue
  fi
  : > "$src"
  if [ -n "$f" ]; then
    # function stubs (return 0: garbage rax breaks callers that check results)
    objdump -T "$f" | awk '$6 != "*UND*" && $NF ~ /^[A-Za-z_]/ {
        s=$NF; sub(/@@.*/,"",s); sub(/@.*/,"",s); print "long "s"(void){return 0;}"}' | sort -u >> "$src" || true
    # data-object stubs
    objdump -T "$f" | awk '$4=="DO" { s=$NF; sub(/@@.*/,"",s); sub(/@.*/,"",s); print "char "s"[64]={0};"}' | sort -u >> "$src" || true

    # collect every version tag any torch object requires from this soname
    tags=""
    for o in "$T"/lib/*.so* "$T"/../_C*.so; do
      [ -f "$o" ] || continue
      objtags=$(objdump -p "$o" 2>/dev/null | awk -v t="$tgt" '
        /required from/ {inblock = (index($0, t) > 0)}
        inblock && $1 ~ /^0x/ && NF >= 4 && $NF ~ /'$TAG_RE'/ {print $NF}')
      tags="$tags $objtags"
    done
    tags=$(echo $tags | tr ' ' '\n' | sort -u || true)
    if [ -n "$(echo $tags | tr -d ' \n')" ]; then
      map="$D/src/${tgt}.map"
      : > "$map"
      for tag in $tags; do
        echo "$tag {" >> "$map"
        echo "  global:" >> "$map"
        objdump -T "$f" | awk '$6 != "*UND*" && $NF ~ /^[A-Za-z_]/ {s=$NF; sub(/@@.*/,"",s); sub(/@.*/,"",s); print "    "s";"}' | sort -u >> "$map"
        echo "  local: *;" >> "$map"
        echo "};" >> "$map"
      done
      gcc -shared -fPIC -Wl,-soname,"$tgt" -Wl,--version-script="$map" -nostdlib -o "$D/$tgt" "$src"
    else
      gcc -shared -fPIC -Wl,-soname,"$tgt" -nostdlib -o "$D/$tgt" "$src"
    fi
  else
    gcc -shared -fPIC -Wl,-soname,"$tgt" -nostdlib -o "$D/$tgt" "$src"
  fi
  echo "  stub: $tgt ($(wc -l < "$src") symbols)"
done

# report any remaining torch .so cuda NEEDEDs not covered by a stub or a real lib
NEED=$(for s in $T/lib/*.so*; do objdump -p "$s" 2>/dev/null | awk '$1=="NEEDED"{print $2}'; done | sort -u | grep -Ei 'cuda|cudnn|nccl|cublas|cufft|curand|cusparse|cupti|nvjit|nvtx|cufile|nvshmem' || true)
for n in $NEED; do
  if [ ! -e "$D/$n" ] && ! ls "$T"/lib/"$n" >/dev/null 2>&1; then
    echo "MISSING stub: $n -- build will fail" >&2
  fi
done
echo "stub dir ready: $D"
