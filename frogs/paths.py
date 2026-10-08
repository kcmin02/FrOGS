"""Repository-relative locations, overridable through environment variables."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("FROGS_DATA_DIR", os.path.join(ROOT, "data"))
RESULTS_DIR = os.environ.get("FROGS_RESULTS_DIR", os.path.join(ROOT, "results"))
FIGURES_DIR = os.environ.get("FROGS_FIGURES_DIR", os.path.join(ROOT, "figures"))
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def data_file(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def figure_path(name: str) -> str:
    return os.path.join(ensure_dir(FIGURES_DIR), name)


def resolve_data_file(path: str | None, default: str | None = None) -> str:
    """Find a data file given as absolute, repo-relative, or by basename in DATA_DIR.

    Checkpoints store the table path of the machine that trained them, so the
    basename fallback lets them load in a fresh clone.
    """
    tried = []
    for cand in (path, default):
        if cand is None:
            continue
        cand = os.path.expanduser(str(cand))
        for p in (cand if os.path.isabs(cand) else os.path.join(ROOT, cand),
                  data_file(os.path.basename(cand))):
            if os.path.exists(p):
                return p
            tried.append(p)
    raise FileNotFoundError("data file not found; tried:\n  " + "\n  ".join(tried))
