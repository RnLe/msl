"""Immutable run manifests (schema ea-operator-v1).

Every v5 archive/result carries one; loaders REFUSE data without an unambiguous
manifest (no guessed band order, no environment-variable physics, no silent defaults).
The manifest hash keys checkpoint reuse: any mismatch aborts instead of mixing runs.
"""
import hashlib
import json
import subprocess

import numpy as np
import yaml

SCHEMA_VERSION = "ea-operator-v1"

REQUIRED = [
    "schema_version",
    "source", "formulation", "geometry", "momentum", "bands", "numerics",
]
REQUIRED_FORMULATION = ["polarization", "spectral_variable", "operator_model"]
REQUIRED_BANDS = ["band_axis_order", "axis_band_ids", "retained_band_ids",
                  "remote_band_ids"]
REQUIRED_GEOMETRY = ["registry_axis_order", "registry_origin"]

OPERATOR_MODELS = (
    "raw_direct_projection_v1",
    "manifest_lowdin_v1",
    "covariant_compact_v1",
    "exact_parent_dispersion_oracle_v1",
)


def git_state(repo):
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", repo, "status", "--porcelain"], text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": True}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode()
                          + str(a.shape).encode()).hexdigest()


def _canonical(obj):
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    return obj


def manifest_hash(m):
    payload = json.dumps(_canonical(m), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def validate(m):
    errors = []
    if m.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version != {SCHEMA_VERSION}")
    for k in REQUIRED:
        if k not in m:
            errors.append(f"missing section: {k}")
    if "formulation" in m:
        for k in REQUIRED_FORMULATION:
            if k not in m["formulation"]:
                errors.append(f"missing formulation.{k}")
        om = m["formulation"].get("operator_model")
        if om is not None and om not in OPERATOR_MODELS:
            errors.append(f"unknown operator_model: {om}")
    if "bands" in m:
        for k in REQUIRED_BANDS:
            if k not in m["bands"]:
                errors.append(f"missing bands.{k}")
        if m["bands"].get("band_axis_order") not in (
                "absolute_eigenvalue_order", "retained_first"):
            errors.append("bands.band_axis_order must be explicit")
    if "geometry" in m:
        for k in REQUIRED_GEOMETRY:
            if k not in m["geometry"]:
                errors.append(f"missing geometry.{k}")
    if errors:
        raise ValueError("invalid manifest: " + "; ".join(errors))
    return True


def save(m, path):
    validate(m)
    m = dict(m)
    m["manifest_hash"] = manifest_hash(
        {k: v for k, v in m.items() if k != "manifest_hash"})
    with open(path, "w") as f:
        yaml.safe_dump(_canonical(m), f, sort_keys=False)
    return m["manifest_hash"]


def load(path):
    with open(path) as f:
        m = yaml.safe_load(f)
    validate(m)
    stored = m.get("manifest_hash")
    actual = manifest_hash({k: v for k, v in m.items() if k != "manifest_hash"})
    if stored != actual:
        raise ValueError(f"manifest hash mismatch: stored {stored} != {actual}")
    return m
