#!/usr/bin/env python3
"""Build exl3_moe_x_ext (kernel-shape variants of exl3_moe) against the installed exllamav3_ext source tree."""
import argparse, hashlib, json, os, shutil, sys, sysconfig, time
from pathlib import Path
from artifact import digest, kernel_digest
EXT_DEFAULT = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
MODULE = "exl3_moe_x_ext"
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ext", default=EXT_DEFAULT); ap.add_argument("--arch", default="121a"); ap.add_argument("--verbose", action="store_true")
ap.add_argument("--install", default="", help="copy the built module into this site-packages dir")
ap.add_argument("--validate-device", action="store_true", help="require CUDA and validate device properties after compiling")
ap.add_argument("--image-id", default="", help="pinned serving image ID, required for --install")
a = ap.parse_args()
if a.install and not a.image_id:
    ap.error("--install requires --image-id for artifact provenance")
import torch  # noqa
from torch.utils.cpp_extension import load
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tree = out / "ext"
if tree.is_dir(): shutil.rmtree(tree)
shutil.copytree(a.ext, tree, ignore=shutil.ignore_patterns("*.so", "__pycache__"))
for n in ("exl3_moe_x.cu", "exl3_moe_x.cuh"): shutil.copyfile(Path(a.src) / n, tree / "quant" / n)
cu13 = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
for var in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    os.environ[var] = cu13 + (":" + os.environ[var] if os.environ.get(var) else "")
t0 = time.time()
mod = load(name=MODULE, sources=[str(tree / "quant" / "exl3_moe_x.cu")],
           extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch=compute_{a.arch},code=sm_{a.arch}", "-Xcudafe", "--diag_suppress=177", "-Xcudafe", "--diag_suppress=20012", "--use_fast_math"],
           extra_include_paths=[str(tree)], build_directory=str(out), verbose=a.verbose)
print(f"built {MODULE} in {time.time()-t0:.0f}s: {mod.__file__}", flush=True)
assert mod.moex_num_variants() == 7
assert all(mod.moex_variant_index(2, k) >= 0 for k in (2, 3, 4))
if a.validate_device:
    if not torch.cuda.is_available():
        raise RuntimeError("--validate-device requires a CUDA device; build success alone is not GPU qualification")
    for v in range(mod.moex_num_variants()):
        info = mod.moex_info(v)
        if info[2] <= 0 or info[4] <= 0:
            raise RuntimeError(f"invalid device occupancy for variant {v}: {info}")
        print(v, mod.moex_name(v), "regs/smem/maxblk/block/sms/local/maxthreads =", info, flush=True)
else:
    print("GPU validation skipped (compile/import only); validate on BOTH nodes before serving", flush=True)
if a.install:
    Path(a.install).mkdir(parents=True, exist_ok=True)
    so = Path(mod.__file__); dest = Path(a.install) / so.name; shutil.copyfile(so, dest); print(f"installed {dest}", flush=True)
if a.install and a.image_id != "in-image-build":
    # A JIT artifact directory is dedicated to this extension. Do not write a
    # generic manifest/helper into site-packages during a full image build.
    source_hash = hashlib.sha256()
    for path in sorted(tree.rglob("*")):
        if path.is_file() and path.suffix in (".h", ".cuh", ".cu", ".cpp"):
            source_hash.update(str(path.relative_to(tree)).encode())
            source_hash.update(path.read_bytes())
    compatibility = {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "python_abi": sysconfig.get_config_var("SOABI"),
                     "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                     "source_sha256": source_hash.hexdigest(), "arch": a.arch}
    manifest = {"file": dest.name, "sha256": digest(dest), "image_id": a.image_id,
                "kernel_sha256": kernel_digest(a.src), "compatibility": compatibility,
                "compatibility_sha256": hashlib.sha256(json.dumps(compatibility, sort_keys=True).encode()).hexdigest()}
    (Path(a.install) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
