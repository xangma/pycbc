#!/usr/bin/env python3
"""Prepare and verify the provenance consumed by JAX parity tools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sealed(value):
    value = dict(value)
    digest = hashlib.sha256(canonical_bytes(value)).hexdigest()
    value["content_sha256"] = digest
    return value


def build_manifest(label, output_path):
    import pycbc

    meta = {
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "pycbc_version": pycbc.__version__,
        "jax_version": metadata.version("jax") if "jax" in sys.modules or metadata.distributions() else None,
    }
    sealed_meta = sealed(meta)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(canonical_bytes(sealed_meta))
    return sealed_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="jax-parity")
    parser.add_argument("--output", default="artifacts/jax_parity_manifest.json")
    args = parser.parse_args()
    build_manifest(args.label, args.output)
    print(f"Wrote manifest for {args.label} to {args.output}")


if __name__ == "__main__":
    main()
