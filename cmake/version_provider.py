"""scikit-build-core metadata provider for DNDSR version.

Reads the base version from the VERSION file and appends git describe
information in PEP 440 format. Commits after the matching release tag are
post-releases; commits toward a newer VERSION are development releases.

Usage in pyproject.toml:
    [project]
    dynamic = ["version"]
    [tool.scikit-build.metadata]
    version.provider = "provider"
    version.provider-path = "cmake"
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _version_from_pkg_info(root: Path) -> str | None:
    """Recover the already-computed version when building from an sdist."""
    try:
        for line in (root / "PKG-INFO").read_text(encoding="utf-8").splitlines():
            if line.startswith("Version: "):
                return line.removeprefix("Version: ").strip()
    except FileNotFoundError:
        pass
    return None


def dynamic_metadata(
    field: str,
    settings: dict[str, Any] | None = None,
) -> str:
    if field != "version":
        msg = f"Only 'version' is supported, got {field!r}"
        raise ValueError(msg)

    root = Path(__file__).resolve().parent.parent
    version_file = root / "VERSION"
    base = version_file.read_text().strip()

    try:
        desc = subprocess.check_output(
            ["git", "describe", "--tags", "--long", "--match", "v*"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # An sdist has no .git directory. Reuse the version written into its
        # core metadata so sdist -> wheel preserves the exact version.
        packaged = _version_from_pkg_info(root)
        if packaged:
            return packaged

        # No git, tags, or package metadata — retain a unique commit hash when
        # possible (for example in a shallow source checkout).
        try:
            short = subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return f"{base}.dev0+g{short}"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return base

    # Parse: v0.0.2-235-gbe407e3
    import re

    m = re.match(r"^v(\d+\.\d+\.\d+)-(\d+)-g([0-9a-f]+)$", desc)
    if not m:
        return _version_from_pkg_info(root) or base

    tag_version = m.group(1)
    distance = int(m.group(2))
    commit = m.group(3)

    if distance == 0 and tag_version == base:
        return base
    if tag_version == base:
        return f"{base}.post{distance}+g{commit}"
    return f"{base}.dev{distance}+g{commit}"


def get_requires_for_dynamic_metadata(
    settings: dict[str, Any] | None = None,
) -> list[str]:
    return []
