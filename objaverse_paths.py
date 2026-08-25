"""objaverse_paths.py — Point the `objaverse` package at ROAHM's pre-downloaded
shared dataset instead of the default ~/.objaverse cache.

Without this, every `objaverse.load_objects(...)` miss falls back to a
one-file-at-a-time HTTPS download (this is what made precompute's Objaverse
fetches slow). Seth has already mirrored the dataset (cloned via hfd.sh, so
it has the raw Hugging Face repo layout: glbs/, metadata/, object-paths.json.gz
directly under the root) on the alexandria.engin.umich.edu NFS export — but
different environments mount that SAME export at different paths:
  - directly at /mnt/workspace/datasets/objaverse/objaverse
  - at /mnt/hostmnt/workspace/datasets/objaverse/objaverse (inside the sventwo
    Docker training container, where /mnt/workspace itself isn't mounted at
    all — only /mnt/hostmnt/workspace/{users,datasets} are)

configure_objaverse() checks both candidates and uses whichever actually
exists, so the same code works in both places without hand-editing paths.

Usage: call configure_objaverse() once, immediately after `import objaverse`,
before any load_* call.
"""

import os

_CANDIDATE_ROOTS = [
    "/mnt/workspace/datasets/objaverse/objaverse",
    "/mnt/hostmnt/workspace/datasets/objaverse/objaverse",
]


def _resolve_objaverse_root() -> str:
    for root in _CANDIDATE_ROOTS:
        if os.path.isdir(root):
            return root
    raise FileNotFoundError(
        "Could not find the shared Objaverse mirror at any known mount path "
        f"({', '.join(_CANDIDATE_ROOTS)}). Falling back to the default "
        "~/.objaverse cache would silently re-download over the network — "
        "check this environment's mounts (e.g. `mount | grep workspace`) and "
        "add the new path to _CANDIDATE_ROOTS above if it's mounted somewhere else."
    )


def configure_objaverse() -> None:
    import objaverse
    root = _resolve_objaverse_root()
    objaverse.BASE_PATH = os.path.dirname(root)
    objaverse._VERSIONED_PATH = root
