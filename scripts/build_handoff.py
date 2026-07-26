"""Build a GPU handoff folder from HEAD, immediately before upload.

Round 2 lost a deliverable to a stale snapshot: `upload_to_drive_2.0/repo_snapshot` was
built from an older HEAD, the `from eval...` sys.path fix landed on the laptop *after* the
A100 session had already run, and the detection cell died on an import that was fixed here
hours earlier. Nothing warned anyone, because a snapshot has no way to say how old it is.

So the snapshot is not assembled by hand any more. This script archives HEAD, records the
commit it came from, and refuses to build from a dirty tree — uncommitted work is invisible
to `git archive`, which is exactly how a "fixed" bug ships broken. The recorded SHA is what
the notebook's preflight cell checks before it spends any A100 time.

    python scripts/build_handoff.py --out upload_to_drive_3.0 --notebook <path>.ipynb
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _assert_clean(allow_dirty: bool) -> None:
    dirty = _git("status", "--porcelain")
    if not dirty:
        return
    message = (
        "Working tree is dirty. `git archive HEAD` would silently omit these changes, so the "
        "snapshot would not contain the code you just wrote:\n"
        + "\n".join(f"    {line}" for line in dirty.splitlines())
    )
    if not allow_dirty:
        raise SystemExit(f"STOP: {message}\n\nCommit first, or pass --allow-dirty if you are sure.")
    print(f"WARNING: {message}\n")


def build(
    out_dir: Path,
    notebook: Path | None,
    extra: list[Path],
    allow_dirty: bool,
    start_here: Path | None = None,
) -> None:
    _assert_clean(allow_dirty)
    sha = _git("rev-parse", "HEAD")
    subject = _git("log", "-1", "--format=%s")
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    snapshot = out_dir / "repo_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir(parents=True)
    # git archive is the only way to get exactly-HEAD without dragging in artifacts/, data/,
    # results/ or a previous handoff nested inside this one (a real bug from Round 1).
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout
    tar_path = out_dir / "_snapshot.tar"
    tar_path.write_bytes(archive)
    shutil.unpack_archive(str(tar_path), str(snapshot), format="tar")
    tar_path.unlink()

    (out_dir / "SNAPSHOT.txt").write_text(
        f"commit: {sha}\nsubject: {subject}\nbuilt_at: {built_at}\n", encoding="utf-8"
    )
    # The notebook's preflight reads this to prove it is running the code it was specified
    # against, so it has to live INSIDE the snapshot too — a SNAPSHOT.txt beside the folder
    # could be refreshed without the folder being rebuilt.
    (snapshot / "SNAPSHOT_SHA.txt").write_text(sha + "\n", encoding="utf-8")

    if notebook:
        shutil.copy2(notebook, out_dir / notebook.name)
    if start_here:
        # Copied under the conventional name: the zero-memory Drive session is told to
        # open START_HERE.md, so the tracked source can be named whatever suits the repo.
        shutil.copy2(start_here, out_dir / "START_HERE.md")
    for path in extra:
        shutil.copy2(path, out_dir / path.name)

    n_files = sum(1 for _ in snapshot.rglob("*") if _.is_file())
    print(f"Handoff built at {out_dir}")
    print(f"  commit      {sha[:12]}  {subject}")
    print(f"  built_at    {built_at}")
    print(f"  snapshot    {n_files} files")
    for item in sorted(out_dir.iterdir()):
        print(f"  - {item.name}")
    print("\nUpload this folder to Drive NOW. If you commit anything else before uploading,")
    print("re-run this script - the snapshot is only as fresh as the moment it was built.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="handoff folder to build (gitignored)")
    ap.add_argument("--notebook", help="A100 notebook to include")
    ap.add_argument("--start-here", help="file to copy in as START_HERE.md")
    ap.add_argument("--extra", nargs="*", default=[], help="briefs / START_HERE to include")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build anyway from a dirty tree (the snapshot will NOT contain uncommitted work)",
    )
    args = ap.parse_args()
    build(
        out_dir=Path(args.out),
        notebook=Path(args.notebook) if args.notebook else None,
        extra=[Path(p) for p in args.extra],
        allow_dirty=args.allow_dirty,
        start_here=Path(args.start_here) if args.start_here else None,
    )


if __name__ == "__main__":
    sys.exit(main())
