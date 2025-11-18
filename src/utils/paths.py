from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
import os
import logging

ABS_HINTS = (":\\", ":/", "\\\\")   # Window drive, URL-like, UNC shares
FORBID_PREFIXES = ("/",)            # Unix absolute root

def get_project_root() -> Path:
    """
    Resolve the MedIm project root directory.

    Resolution order
    ----------------
    1) Environment variable (e.g. MEDIM_PROJ_ROOT)
    2) Traverse upwards from this file to find a folder containing 'data' and 'src'
    3) Fallback to current working directory

    Returns
    -------
    Path
        Project root path (absolute, resolved).
    """
    logger = logging.getLogger(__name__)
    env_var_name = "MEDIM_PROJ_ROOT"
    root_env = os.getenv(env_var_name)

    if root_env:
        root_path = Path(root_env).expanduser().resolve()
        if not root_path.exists():
            logger.warning("[PATH] Env %s=%r does not exist, ignoring.",env_var_name, root_env )
        else:
            return Path(root_env).resolve()
    
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        data_dir = parent / "data"
        src_dir = parent / "src"

        if data_dir.is_dir() and src_dir.is_dir():
            return parent
    
    logger.warning(
        "[paths] Could not infer project root from env or layout, "
        "falling back to current working directory."
    )
    
    return Path.cwd().resolve()


def looks_absolute(path_str: Union[Path, str]) -> bool:
    """
    Heuristic detector of absolute path strings.

    Parameters
    ----------
    path_str : str

    Returns
    -------
    bool
    """
    s = str(path_str or "").strip()
    
    if s.startswith(FORBID_PREFIXES):
        return True
    
    return any(hint in s for hint in ABS_HINTS)

def to_relpath(path_like: str | Path, root: Optional[Path] = None) -> str:
    """
    Convert an absolute path into a POSIX-style relative path w.r.t project root.

    Parameters
    ----------
    path_like : str | Path
        Path to convert.

    root      : Optional[Path]
        Base directory (default: project root via `get_project_root()`).

    Returns
    -------
    str
        POSIX relative path (using "/").
    """
    base = Path(root or get_project_root()).resolve()
    p_in = Path(path_like).expanduser().resolve()

    if not _is_relative_to(p_in, base):
        raise ValueError(f"Path {p_in} is not under project root {base}")
    
    rel = p_in.relative_to(base)
    return rel.as_posix()

def _is_relative_to(child: Path, parent: Path) -> bool:
    """
    Local helper tương tự Path.is_relative_to (Py>=3.9).

    Returns
    -------
    bool
        True if p is inside base (or equal), False otherwise.
    """

    try:
        _ = child.relative_to(parent)
        return True
    except ValueError:
        return False

def safe_join_rel(*parts: Union[str, Path], root: Optional[Path] = None) -> Path:
    """
    Join relative parts under the project root and return a normalized, resolved Path.

    Safety guarantees
    -----------------
    - Rejects absolute or URL-like inputs (Windows drive, UNC, scheme://).
    - Prevents parent-traversal escapes (...).
    - Ensures the resolved path remains inside the project root.

    Parameters
    ----------
    parts : str | Path
        Path components (relative to project root).
    root  : Optional[Path]
        Project root (default to get_project_root()).
    
    Returns
    -------
    Path
        Resolved path strictly inside project root.

    Raises
    ------
    ValueError
        If an input part is absolute/URL-like, contains traversal,
        or the resolved path escapes the project root.
    """
    base = Path(root or get_project_root()).resolve()
    clean_parts: List[Path] = []

    for part in parts:
        p = Path(part)

        if looks_absolute(p):
            raise ValueError(f"Absolute/URL-like path not allowed: {part!r}")
        
        # prevent sneaky parent traversal in components
        if any(seg == ".." for seg in p.parts):
            raise ValueError(f"Parent-traversal '..' not allowed: {part!r}")
        
        clean_parts.append(p)
    
    candidate = (base.joinpath(*clean_parts)).resolve()

    # final guard: must still be inside base after resolution/symlinks
    if not _is_relative_to(candidate, base):
        raise ValueError(f"Resolved path escaped project root: {candidate} (root={base})")
    
    return candidate