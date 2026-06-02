"""The ``rb://`` fsspec filesystem for Rebase (rebase.energy).

``rb://`` is a thin, **Hugging Face-backed** scheme: storage lives on the HF Hub
and the ``rb://`` prefix signals that a given HF dataset/model repo is
*Rebase-compatible*. Registering it via :mod:`fsspec` is exactly how Hugging
Face makes ``hf://`` work — once registered, any tool that goes through fsspec
(pandas, polars, duckdb, pyarrow, ...) understands the scheme.

    >>> import emflow            # registers the ``rb`` protocol on import
    >>> import pandas as pd
    >>> df = pd.read_parquet("rb://dataset/acme/wind-farm-a/data.parquet")

URI layout — ``kind`` maps to the HF ``repo_type`` and the rest is the repo id
plus an in-repo path::

    rb://dataset/<owner>/<repo>/<path>   ->  hf://datasets/<owner>/<repo>/<path>
    rb://model/<owner>/<repo>/<path>     ->  hf://<owner>/<repo>/<path>

Compatibility: on first access to a repo, the repo is checked for a Rebase
marker — a ``rebase.yaml``/``rebase.yml``/``rebase.json`` file at the repo root,
or a ``rebase``/``rebase-compatible`` tag. Repos without a marker raise
:class:`RebaseIncompatibleError`. Pass ``skip_compat_check=True`` (filesystem
kwarg) to bypass.
"""

from __future__ import annotations

from fsspec import AbstractFileSystem, filesystem, register_implementation
from fsspec.utils import stringify_path

#: rb:// kind -> HF repo_type.
KIND_TO_REPO_TYPE = {"dataset": "dataset", "model": "model"}

#: Repo-root files that mark a repo as Rebase-compatible.
MARKER_FILES = ("rebase.yaml", "rebase.yml", "rebase.json")

#: HF tags that mark a repo as Rebase-compatible.
MARKER_TAGS = ("rebase", "rebase-compatible")


class RebaseIncompatibleError(Exception):
    """Raised when an ``rb://`` repo lacks a Rebase-compatibility marker."""


class RebaseFileSystem(AbstractFileSystem):
    """fsspec filesystem backing the ``rb://`` scheme with Hugging Face storage.

    Read-only. Translates ``rb://<kind>/<owner>/<repo>/<path>`` to the matching
    ``hf://`` location, verifies Rebase-compatibility once per repo, and
    delegates byte IO to :class:`huggingface_hub.HfFileSystem`.

    Parameters
    ----------
    token:
        HF token for private repos (falls back to ``HF_TOKEN`` / cached login).
    endpoint:
        Custom HF Hub endpoint.
    skip_compat_check:
        If True, skip the Rebase-marker verification (treat ``rb://`` as a pure
        alias for ``hf://``).
    """

    protocol = "rb"
    root_marker = ""

    def __init__(self, *args, token=None, endpoint=None,
                 skip_compat_check=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._token = token
        self._skip_compat = skip_compat_check
        self._compat_cache: dict[tuple[str, str], bool] = {}
        hf_kwargs = {}
        if token is not None:
            hf_kwargs["token"] = token
        if endpoint is not None:
            hf_kwargs["endpoint"] = endpoint
        try:
            self._hf = filesystem("hf", **hf_kwargs)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "rb:// requires huggingface_hub for Hugging Face-backed storage. "
                "Install it with `pip install huggingface_hub` (or "
                "`pip install emflow[submissions]`)."
            ) from exc

    # -- parsing / translation ---------------------------------------------

    @classmethod
    def _strip_protocol(cls, path) -> str:
        """Normalise ``rb://dataset/a/b`` (or ``rb:/...`` / ``rb:...``) to ``dataset/a/b``."""
        path = stringify_path(path)
        for prefix in ("rb://", "rb:/", "rb:"):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        return path.strip("/")

    def _parse(self, path) -> tuple[str, str, str]:
        """Return ``(repo_type, repo_id, in_repo_path)`` for an rb path."""
        rel = self._strip_protocol(path)
        if not rel:
            raise FileNotFoundError(
                "rb:// requires rb://<kind>/<owner>/<repo>[/<path>]"
            )
        parts = rel.split("/")
        kind = parts[0]
        if kind not in KIND_TO_REPO_TYPE:
            raise FileNotFoundError(
                f"rb://{rel}: unknown kind {kind!r} "
                f"(expected one of {tuple(KIND_TO_REPO_TYPE)})"
            )
        if len(parts) < 3:
            raise FileNotFoundError(
                f"rb://{rel}: expected rb://{kind}/<owner>/<repo>[/<path>]"
            )
        repo_id = f"{parts[1]}/{parts[2]}"
        in_repo = "/".join(parts[3:])
        return KIND_TO_REPO_TYPE[kind], repo_id, in_repo

    @staticmethod
    def _hf_path(repo_type: str, repo_id: str, in_repo: str) -> str:
        """Build the equivalent ``hf://`` path."""
        base = repo_id if repo_type == "model" else f"{repo_type}s/{repo_id}"
        return f"hf://{base}/{in_repo}" if in_repo else f"hf://{base}"

    # -- compatibility marker ----------------------------------------------

    def _is_rebase_compatible(self, repo_type: str, repo_id: str) -> bool:
        from huggingface_hub import HfApi

        api = HfApi(token=self._token)
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type,
                             files_metadata=False)
        tags = set(getattr(info, "tags", None) or [])
        if tags.intersection(MARKER_TAGS):
            return True
        siblings = getattr(info, "siblings", None) or []
        names = {s.rfilename for s in siblings}
        return any(marker in names for marker in MARKER_FILES)

    def _check_compat(self, repo_type: str, repo_id: str) -> None:
        if self._skip_compat:
            return
        key = (repo_type, repo_id)
        if key not in self._compat_cache:
            self._compat_cache[key] = self._is_rebase_compatible(*key)
        if not self._compat_cache[key]:
            raise RebaseIncompatibleError(
                f"rb://{repo_type}/{repo_id} is not Rebase-compatible: the repo "
                f"has none of the marker files {MARKER_FILES} and none of the "
                f"tags {MARKER_TAGS}. Use hf:// directly, or pass "
                f"skip_compat_check=True to bypass."
            )

    def _resolve(self, path) -> str:
        repo_type, repo_id, in_repo = self._parse(path)
        self._check_compat(repo_type, repo_id)
        return self._hf_path(repo_type, repo_id, in_repo)

    # -- fsspec API ---------------------------------------------------------

    def _open(self, path, mode="rb", **kwargs):
        if any(c in mode for c in "wa+"):
            raise NotImplementedError("rb:// is read-only")
        return self._hf.open(self._resolve(path), mode=mode)

    def info(self, path, **kwargs):
        info = dict(self._hf.info(self._resolve(path)))
        info["name"] = self._strip_protocol(path)
        return info

    def ls(self, path, detail=True, **kwargs):
        return self._hf.ls(self._resolve(path), detail=detail, **kwargs)

    def exists(self, path, **kwargs):
        try:
            resolved = self._resolve(path)
        except (FileNotFoundError, RebaseIncompatibleError, NotImplementedError):
            return False
        return self._hf.exists(resolved)


def register(clobber: bool = True) -> None:
    """Register :class:`RebaseFileSystem` under the ``rb`` protocol.

    Called automatically on ``import emflow``. Does not import
    ``huggingface_hub`` — that happens lazily when ``rb://`` is first used.
    """
    register_implementation("rb", RebaseFileSystem, clobber=clobber)
