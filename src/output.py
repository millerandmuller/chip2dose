"""Writing result files to disk.

Every result is written to a temporary name in the destination's own directory and then renamed into
place. An interrupted run therefore leaves either the previous file or the complete new one, never a
half-written figure or table that would look like a result. The temporary name starts with a dot and
ends in `.part`, and is removed if the write raises; a process killed outright (SIGKILL) can leave
one behind, but no result file is ever partially overwritten.

Writers under `validation/` deliberately do not use this module: those files are fingerprinted at the
pre-registration commit and are not touched here.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def atomic_write(path: Path, write: Callable[[Path], None]) -> Path:
    """Run `write` against a temporary path in `path`'s directory, then rename it over `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    os.close(handle)
    tmp = Path(name)
    try:
        write(tmp)
        os.replace(tmp, path)  # atomic within one filesystem
    except BaseException:  # including KeyboardInterrupt: the case this module exists for
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write(path, lambda tmp: tmp.write_text(text, encoding=encoding))
