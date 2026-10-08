"""Reading and writing importance-sampling sweep files.

A sweep npz holds, on a (n_mu, n_T) grid of conditions c = (dmu, T):

    dmu (n_mu,), T (n_T,)
    M, logZ, logZ_se, logZ_raw, x, x_se, ess          (n_mu, n_T)
    A_samples, N_samples                              (n_mu, n_T, M)   optional
    sigma_packed                                      (n_mu, n_T, M, ceil(N/8)) uint8, optional
    meta_json                                         JSON string; meta["N"] is the site count

`N_samples` counts sites with sigma = +1 (Ag for AgPd, Au for CuAu).
"""
from __future__ import annotations

import json

import numpy as np

# Older sweep files used these names.
LEGACY_KEYS = {
    "x_Ag": "x",
    "x_Ag_se": "x_se",
    "x_Au": "x",
    "N_Ag_samples": "N_samples",
    "N_plus_samples": "N_samples",
}
SKIP_KEYS = {"logZ_0_anchor", "logZ_diff_kappa2", "meta_json"}


def load_sweep(path: str, keys=None) -> dict:
    """Load a sweep npz into a dict with canonical key names and a parsed `meta`."""
    out = {}
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        for name in z.files:
            if name in SKIP_KEYS:
                continue
            canon = LEGACY_KEYS.get(name, name)
            if canon in out or (keys is not None and canon not in keys):
                continue
            out[canon] = z[name]
    out["meta"] = meta
    if "N" not in meta and "N_samples" in out:
        meta["N"] = int(np.nanmax(out["N_samples"]))
    return out


def save_sweep(path: str, arrays: dict, meta: dict) -> None:
    payload = dict(arrays)
    payload["meta_json"] = np.array(json.dumps(meta), dtype=object)
    np.savez(path, **payload)


def unpack_sigma(packed: np.ndarray, n_sites: int) -> np.ndarray:
    """Bit-packed {0,1} occupations -> int8 spins in {-1, +1}."""
    bits = np.unpackbits(packed, axis=-1)[..., :n_sites].astype(np.int8)
    return bits * 2 - 1
