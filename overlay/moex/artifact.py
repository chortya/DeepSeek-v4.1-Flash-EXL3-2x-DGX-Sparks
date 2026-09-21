#!/usr/bin/env python3
"""Validate a local JIT artifact without importing Torch or touching a GPU."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel_digest(source):
    root = Path(source)
    return hashlib.sha256(b"".join((root / n).read_bytes() for n in
                          ("exl3_moe_x.cu", "exl3_moe_x.cuh"))).hexdigest()


def verify(directory, image_id, expected_kernel):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    name = manifest["file"]
    if Path(name).name != name or not name.startswith("exl3_moe_x_ext") or not name.endswith(".so"):
        raise ValueError("invalid MoE-X artifact filename")
    if manifest["image_id"] != image_id:
        raise ValueError("MoE-X artifact was built for a different image; rebuild against the pinned serving image")
    if manifest["kernel_sha256"] != expected_kernel:
        raise ValueError("MoE-X artifact kernel source is stale")
    if manifest["sha256"] != digest(root / name):
        raise ValueError("MoE-X artifact checksum mismatch")
    return manifest["compatibility_sha256"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("image_id")
    ap.add_argument("kernel_sha256")
    a = ap.parse_args()
    try:
        print(verify(a.directory, a.image_id, a.kernel_sha256))
    except (OSError, ValueError, KeyError) as exc:
        ap.exit(2, f"MoE-X artifact preflight failed at {a.directory}: {exc}\n")
