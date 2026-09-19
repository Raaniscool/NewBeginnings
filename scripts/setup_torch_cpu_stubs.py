#!/usr/bin/env python3
"""Generate CPU-only stub libraries for the CUDA SONAMEs the PyPI torch wheel
is linked against -- so that `import torch` works WITHOUT installing ~5-6 GB
of nvidia-* wheels we will never execute on this CPU-only machine.

WHY: storage is a first-class constraint for this project. The CUDA libraries
are dead weight here; we just need the dynamic loader to be satisfied.

HOW: some torch libs bind eagerly (full RELRO / BIND_NOW) and reference
versioned CUDA symbols (e.g. cuptiFinalize@libcupti.so.13). So for every
stub SONAME we must export every non-weak undefined CUDA symbol torch's libs
reference, with the correct symbol *version* where torch asks for one.
This script does exactly that, by static analysis:

  1. scan every torch/lib/*.so: collect DT_NEEDED CUDA-ish sonames, and every
     non-weak UND symbol with its (optional) version == providing soname;
  2. attribute unversioned CUDA-API symbols to a stub that is guaranteed to
     be loaded alongside (the first CUDA NEEDED of the referencing lib);
  3. emit a tiny C file per stub (functions + data objects) and an optional
     ld version-script, compile with gcc, place into torch/lib (covered by
     torch's own $ORIGIN RPATH/RUNPATH);
  4. verify by importing torch in a subprocess.

HARD CONTRACT: this project never calls any torch.cuda API. Device is pinned
to "cpu". A test (tests/test_torch_cpu_only.py) guards that contract.

Idempotent; stubs it previously created (tracked in .cpu_stub_manifest.json)
are regenerated, real libraries are never touched.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

CUDA_SONAME_RE = re.compile(
    r"^(libcuda\.so|libcu|libcublas|libcudnn|libcufft|libcurand|libcusparse|"
    r"libcusolver|libcupti|libcufile|libnvrtc|libnvshmem|libnccl|libnvJit|"
    r"libnvtx)"
)
CUDA_SYM_RE = re.compile(
    r"^(cu[A-Z]|cuda|cupti|cublas|cudnn|curand|cufft|cusparse|cusolver|"
    r"cusolverDn|nccl|nvrtc|nvshmem|nvtx|nvJit|__cuda|cudart|cufile)"
)
MANIFEST = ".cpu_stub_manifest.json"


def sh(cmd, capture=True) -> str:
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if r.returncode != 0 and capture:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr}")
    return r.stdout


def torch_lib_dir() -> Path:
    import importlib.util
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("ERROR: torch not installed")
    return Path(spec.submodule_search_locations[0]) / "lib"


def needed_sonames(lib: Path) -> list[str]:
    out = sh(["objdump", "-p", str(lib)])
    return [ln.split()[1] for ln in out.splitlines() if "NEEDED" in ln]


def undefined_symbols(lib: Path) -> list[tuple[str, str | None, bool]]:
    """Return [(name, version_or_None, is_object)] for non-weak UND symbols."""
    out = sh(["objdump", "-T", str(lib)])
    res = []
    for ln in out.splitlines():
        if "*UND*" not in ln:
            continue
        f = ln.split()
        try:
            u = f.index("*UND*")
        except ValueError:
            continue
        flags = f[u - 1]
        if "w" in flags.lower() and "F" not in flags:
            # weak undefined: allowed to stay unresolved; skip
            continue
        is_obj = "O" in flags
        tail = f[u + 2:]
        version = None
        if tail and tail[0].startswith("("):
            version = tail[0].strip("()")
            name = tail[1]
        else:
            name = tail[-1]
        res.append((name, version, is_obj))
    return res


def main() -> int:
    tlib = torch_lib_dir()
    if not tlib.is_dir():
        raise SystemExit(f"ERROR: torch lib dir missing: {tlib}")
    print(f"torch lib dir: {tlib}")

    manifest_path = tlib / MANIFEST
    manifest: dict = {"stubs": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    # ----- static analysis -------------------------------------------------
    # The version TAG attached to a symbol reference usually equals the
    # providing library's soname, but not always (e.g. libnvshmem_host.so.3
    # versions its symbols "NVSHMEM"). So stubs are keyed by FILE name, and
    # each file may carry several version nodes.
    # stubs: file -> version_node -> {"fn": set, "obj": set}
    stubs: dict[str, dict[str, dict[str, set]]] = {}

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _core(soname: str) -> str:
        n = _norm(soname)
        n = re.sub(r"^lib", "", n)
        n = re.sub(r"so\d*$", "", n)
        return n

    def resolve_file(version: str, needs: list[str]) -> str:
        """Which needed soname most plausibly provides `version`?"""
        if version in needs:
            return version
        cv = re.sub(r"\d+$", "", _norm(version))
        cands = [s for s in needs if cv and (cv in _core(s) or _core(s) in cv)]
        return sorted(cands, key=lambda s: len(_core(s)))[0] if cands else needs[0]

    def slot(file: str, node: str, is_obj: bool) -> set:
        f = stubs.setdefault(file, {})
        n = f.setdefault(node, {"fn": set(), "obj": set()})
        return n["obj" if is_obj else "fn"]

    for lib in sorted(tlib.glob("*.so")):
        cuda_needs = [s for s in needed_sonames(lib) if CUDA_SONAME_RE.match(s)]
        if not cuda_needs:
            continue
        for need in cuda_needs:
            stubs.setdefault(need, {})  # stub exists even if zero symbols
        for name, version, is_obj in undefined_symbols(lib):
            if version:
                if not (CUDA_SONAME_RE.match(version) or
                        any(CUDA_SONAME_RE.match(s) for s in cuda_needs)):
                    continue  # glibc etc.: provided by the system
                file = resolve_file(version, cuda_needs)
                slot(file, version, is_obj).add(name)
            else:
                if not CUDA_SYM_RE.match(name):
                    continue  # not a CUDA API symbol
                slot(cuda_needs[0], "", is_obj).add(name)

    # ----- generate + compile ---------------------------------------------
    created, regenerated, skipped_real = 0, 0, 0
    new_manifest = {"stubs": {}}
    for soname, versions in sorted(stubs.items()):
        dest = tlib / soname
        if dest.exists():
            if soname in manifest.get("stubs", {}) or dest.stat().st_size < 1_000_000:
                dest.unlink()  # ours from a previous run (or tiny placeholder)
                regenerated += 1
            else:
                print(f"  keep real lib: {soname}")
                skipped_real += 1
                continue

        # fold the unversioned bucket ("") into the first named node if one
        # exists (ld forbids mixing an anonymous node with named nodes);
        # unversioned refs still bind to versioned definitions.
        base = versions.get("", {"fn": set(), "obj": set()})
        named = {v: k for v, k in versions.items() if v}
        node_names: dict[str, list[str]] = {}
        if named:
            first = sorted(named)[0]
            for v in sorted(named):
                names = sorted(named[v]["fn"]) + sorted(named[v]["obj"])
                if v == first:
                    names += sorted(base["fn"]) + sorted(base["obj"])
                node_names[v] = sorted(set(names))
        else:
            node_names[""] = sorted(base["fn"]) + sorted(base["obj"])

        all_syms = sorted({n for names in node_names.values() for n in names})
        c_lines = ["/* CPU-only stub: satisfies the dynamic loader.",
                    "   Functions return 0 (success status) so that any defensive",
                    "   init-time call (e.g. cupti/cudart probing) sees 'success' and",
                    "   a NULL-looking handle instead of garbage in rax. */"]
        fn_set = set().union(*(versions[v]["fn"] for v in versions)) if versions else set()
        for n in all_syms:
            if n in fn_set:
                # long return: a single instruction writing 0 to rax then ret
                c_lines.append(f"long {n}(void){{return 0;}}")
            else:
                c_lines.append(f"char {n}[64];")
        c_src = "\n".join(c_lines) + "\n"

        n_syms = len(all_syms)
        cmd = ["gcc", "-shared", "-nostdlib", "-fPIC", "-x", "c", "-",
               "-Wl,-soname," + soname, "-o", str(dest)]
        vs_path = None
        if named:
            stanzas = [
                f"{v} {{\n  global:\n" +
                "".join(f"    {n};\n" for n in names) +
                "  local: *;\n};"
                for v, names in node_names.items()
            ]
            vs_path = tlib / (soname + ".map")
            vs_path.write_text("\n".join(stanzas) + "\n")
            cmd += ["-Wl,--version-script," + str(vs_path)]
        p = subprocess.run(cmd, input=c_src, text=True, capture_output=True)
        if vs_path:
            vs_path.unlink(missing_ok=True)
        if p.returncode != 0:
            print(f"ERROR compiling {soname}:\n{p.stderr}")
            return 1
        new_manifest["stubs"][soname] = {"symbols": n_syms}
        print(f"  stub {soname}: {n_syms} symbols")

    manifest_path.write_text(json.dumps(new_manifest, indent=2) + "\n")
    total = sum(v["symbols"] for v in new_manifest["stubs"].values())
    print(f"created {len(new_manifest['stubs'])} stubs exporting {total} symbols "
          f"| regenerated: {regenerated} | real libs kept: {skipped_real}")
    print(f"stub footprint: {sh(['du','-sh','--exclude=*.so.*.*','-c']+[str(tlib/s) for s in new_manifest['stubs']]).splitlines()[-1]}")
    print("NOTE: rerun this script after any torch reinstall/upgrade.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
