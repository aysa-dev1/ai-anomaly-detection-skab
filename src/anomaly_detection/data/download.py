from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DownloadResult:
    repo_url: str
    target_dir: Path
    action: str # possible: cloned, updated, skipped


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def ensure_git_available() -> None:
    if shutil.which("git") is None:
        raise RuntimeError(
            "git is not available on Path. Check installation of git and retry"
        )
    

def clone_or_pull_repo(repo_url: str, target_dir: Path) -> DownloadResult:
    """
    Clone data from GitHub repo into target_dir, or pull changes if already cloned
    """

    ensure_git_available()
    target_dir = target_dir.resolve()

    # case target dir does not exists: clone the git repo
    if not target_dir.exists():
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--depth", "1", repo_url, str(target_dir)])
        action = "cloned"
    else:
        # case target dir exists: pull changes
        git_dir = target_dir / ".git"
        if not git_dir.exists():
            raise RuntimeError(
                f"Target dir exists but is not a git repo: {target_dir}\n"
                "Please remove/rename it or choose a different raw_dir."
            )

        # pulling
        _run(["git", "pull", "--ff-only"], cwd=target_dir)
        action = "updated"

        
    return DownloadResult(repo_url=repo_url, target_dir=target_dir, action=action)