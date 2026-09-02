"""
Read files from a Databricks Unity Catalog (UC) Volume into pandas — locally,
with no Spark and no running cluster.

How it works
------------
Uses the ``databricks-sdk`` Files API, which streams the bytes of a file that
lives on a UC Volume (``/Volumes/<catalog>/<schema>/<volume>/...``) over HTTPS.
We read those bytes straight into pandas via an in-memory buffer, so nothing is
staged to local disk unless you explicitly call :func:`download`.

Auth
----
Re-uses your existing ``~/.databrickscfg`` profile — no tokens in code. The
default profile is ``adb-3525355376223725`` (the workspace that hosts the
``pds_feu_931272_dev`` catalog); override per-call with ``profile=...`` or
globally with the ``DATABRICKS_CONFIG_PROFILE`` env var. If the profile's
OAuth token has expired, refresh it once with::

    databricks auth login --profile adb-3525355376223725

Spark-output directories
------------------------
A Spark/Delta write produces a *directory* of ``part-*.snappy.parquet`` files
plus ``_SUCCESS`` / ``_committed_*`` / ``_started_*`` commit markers. Point
:func:`read_parquet` at that directory and it reads every data part (skipping
the underscore/dot markers) and concatenates them into one DataFrame — the same
result ``spark.read.parquet(dir)`` would give you.

Typical use
-----------
    from utils import uc_io

    BASE = "/Volumes/pds_feu_931272_dev/data_science_team/uk_run/OUTPUTS"

    uc_io.ls(BASE)                                  # list a volume directory
    df  = uc_io.read_parquet(f"{BASE}/TF_DIQ_combined_2026-15")   # whole dataset
    one = uc_io.read_parquet(f"{BASE}/TF_DIQ_combined_2026-15", max_parts=1)  # quick peek
    key = uc_io.read_csv(f"{BASE}/diq_key_lookup_202608.csv")     # a CSV file
    uc_io.download(f"{BASE}/diq_key_lookup_202608.csv", "local.csv")  # stage to disk
"""

from __future__ import annotations

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Workspace that hosts the pds_feu_931272_dev catalog. Env var wins so callers
# can point at another workspace without editing code.
DEFAULT_PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE") or "adb-3525355376223725"

_PARQUET_SUFFIXES = (".parquet", ".pq")


def _norm(path: str) -> str:
    """Accept ``dbfs:/Volumes/...`` (CLI style) or ``/Volumes/...`` (Files-API
    style) and return the plain ``/Volumes/...`` form the Files API expects."""
    if path.startswith("dbfs:"):
        path = path[len("dbfs:"):]
    return path.rstrip("/")


@lru_cache(maxsize=8)
def _client(profile: str):
    """Cached WorkspaceClient per profile. Import is local so the module loads
    even when databricks-sdk isn't installed (helpful error only on first use)."""
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as exc:  # pragma: no cover - install-time guard
        raise ImportError(
            "databricks-sdk is required for utils.uc_io. Install it with:\n"
            "    .venv/bin/pip install --index-url https://pypi.org/simple databricks-sdk"
        ) from exc
    return WorkspaceClient(profile=profile)


def ls(volume_path: str, *, profile: str = DEFAULT_PROFILE, long: bool = False):
    """List the contents of a UC Volume directory.

    Parameters
    ----------
    volume_path : str
        Directory on a Volume, e.g. ``/Volumes/cat/schema/vol/sub`` (``dbfs:``
        prefix tolerated).
    long : bool
        If True, return the raw ``DirectoryEntry`` objects (``.name``,
        ``.path``, ``.is_directory``, ``.file_size``, ``.last_modified``).
        If False (default), return a sorted list of entry names.
    """
    w = _client(profile)
    entries = list(w.files.list_directory_contents(_norm(volume_path)))
    if long:
        return entries
    return sorted(e.name.rstrip("/") for e in entries)


def _parquet_parts(volume_path: str, w) -> List[str]:
    """Resolve ``volume_path`` to a list of parquet file paths.

    A single ``*.parquet`` file -> ``[that file]``. A directory -> every
    ``part-*.parquet`` inside it, skipping ``_SUCCESS`` / ``_committed_*`` /
    ``_started_*`` markers and any nested directories.
    """
    vp = _norm(volume_path)
    if vp.lower().endswith(_PARQUET_SUFFIXES):
        return [vp]
    entries = list(w.files.list_directory_contents(vp))
    parts = [
        e.path for e in entries
        if not e.is_directory
        and not e.name.startswith(("_", "."))
        and e.name.lower().endswith(_PARQUET_SUFFIXES)
    ]
    return sorted(parts)


def _download_bytes(file_path: str, w) -> bytes:
    """Stream a single Volume file into memory as bytes."""
    resp = w.files.download(_norm(file_path))
    return resp.contents.read()


def read_parquet(
    volume_path: str,
    *,
    profile: str = DEFAULT_PROFILE,
    columns: Optional[List[str]] = None,
    max_parts: Optional[int] = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Read a parquet file — or a whole Spark-output directory of parts — from a
    UC Volume into a single pandas DataFrame.

    Parameters
    ----------
    volume_path : str
        A ``*.parquet`` file OR a directory containing ``part-*.parquet`` files.
    columns : list[str], optional
        Column projection — pushed down to each part for a smaller/faster read.
    max_parts : int, optional
        Read only the first N parts (sorted by name). Handy for a quick schema
        peek without pulling the full ~140 MB dataset.
    max_workers : int
        Parts are downloaded concurrently (each part is an independent HTTPS
        GET). Default 8.

    Raises
    ------
    FileNotFoundError
        If no parquet parts are found at ``volume_path``.
    """
    w = _client(profile)
    parts = _parquet_parts(volume_path, w)
    if max_parts is not None:
        parts = parts[:max_parts]
    if not parts:
        raise FileNotFoundError(
            f"No parquet parts found at {volume_path!r}. "
            "Is it a Volume path to a parquet file or a Spark-output directory?"
        )

    def _read_one(p: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(_download_bytes(p, w)), columns=columns)

    if len(parts) == 1:
        return _read_one(parts[0])

    logger.info("[uc_io] reading %d parquet parts from %s", len(parts), volume_path)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(parts))) as ex:
        frames = list(ex.map(_read_one, parts))
    return pd.concat(frames, ignore_index=True)


def read_csv(volume_path: str, *, profile: str = DEFAULT_PROFILE, **csv_kwargs) -> pd.DataFrame:
    """Read a single CSV file from a UC Volume into pandas. Extra kwargs are
    forwarded to ``pd.read_csv`` (e.g. ``dtype=``, ``thousands=','``)."""
    w = _client(profile)
    data = _download_bytes(volume_path, w)
    return pd.read_csv(io.BytesIO(data), **csv_kwargs)


def download(volume_path: str, local_path: str, *, profile: str = DEFAULT_PROFILE) -> str:
    """Download a single Volume file to a local path (creates parent dirs).
    Returns ``local_path``. Use when a tool needs a real file on disk."""
    w = _client(profile)
    data = _download_bytes(volume_path, w)
    parent = os.path.dirname(os.path.abspath(local_path))
    os.makedirs(parent, exist_ok=True)
    with open(local_path, "wb") as fh:
        fh.write(data)
    logger.info("[uc_io] downloaded %s -> %s (%d bytes)", volume_path, local_path, len(data))
    return local_path


__all__ = [
    "DEFAULT_PROFILE",
    "ls",
    "read_parquet",
    "read_csv",
    "download",
]


if __name__ == "__main__":
    # Smoke test: `python utils/uc_io.py [volume_dir]` lists a directory.
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else (
        "/Volumes/pds_feu_931272_dev/data_science_team/uk_run/OUTPUTS"
    )
    print(f"Listing {target} (profile={DEFAULT_PROFILE}):")
    for name in ls(target):
        print(f"  {name}")
