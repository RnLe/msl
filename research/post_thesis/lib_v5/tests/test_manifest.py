"""Manifest contract tests: refuse ambiguity, detect tamper."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib_v5 import manifest as mf  # noqa: E402


def _minimal():
    return {
        "schema_version": mf.SCHEMA_VERSION,
        "source": {"msl_commit": "x", "blaze_native_sha256": "y"},
        "formulation": {"polarization": "TM", "spectral_variable": "lambda",
                        "operator_model": "raw_direct_projection_v1"},
        "geometry": {"registry_axis_order": ["s1", "s2"],
                     "registry_origin": "node"},
        "momentum": {"k0_primitive_fractional": [0.5, 0.0]},
        "bands": {"band_axis_order": "absolute_eigenvalue_order",
                  "axis_band_ids": [0, 1, 2, 3],
                  "retained_band_ids": [1],
                  "remote_band_ids": [0, 2, 3]},
        "numerics": {"dtype": "complex128"},
    }


def test_roundtrip_and_hash():
    m = _minimal()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.yaml")
        h = mf.save(m, p)
        m2 = mf.load(p)
        assert m2["manifest_hash"] == h


def test_tamper_detected():
    m = _minimal()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.yaml")
        mf.save(m, p)
        txt = open(p).read().replace("TM", "TE")
        open(p, "w").write(txt)
        try:
            mf.load(p)
            raise AssertionError("tampered manifest accepted")
        except ValueError:
            pass


def test_missing_band_order_refused():
    m = _minimal()
    del m["bands"]["band_axis_order"]
    try:
        mf.validate(m)
        raise AssertionError("ambiguous band order accepted")
    except ValueError:
        pass


def test_unknown_operator_model_refused():
    m = _minimal()
    m["formulation"]["operator_model"] = "EA"
    try:
        mf.validate(m)
        raise AssertionError("vague operator model accepted")
    except ValueError:
        pass


def test_array_hash_sensitivity():
    a = np.arange(12, dtype=float).reshape(3, 4)
    h1 = mf.array_hash(a)
    b = a.copy()
    b[1, 2] += 1e-15
    assert mf.array_hash(b) != h1
    assert mf.array_hash(a.astype(np.float32)) != h1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
