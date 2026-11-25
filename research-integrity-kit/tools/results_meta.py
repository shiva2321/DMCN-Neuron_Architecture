"""Run-metadata helper for results files (GEMINI.md §6).

Every results JSON that supports a claim must carry a top-level ``"_meta"`` block with the
git commit, experiment id, seed list, full config, command, and UTC timestamp.

Usage in an experiment script::

    sys.path.insert(0, ".integrity"); from results_meta import build_meta, write_results
    meta = build_meta(exp_id="my-exp-v2", seeds=list(range(100, 120)), config=cfg)
    write_results("results/my-exp-v2/results.json", payload, meta)

Gate (``python .integrity/results_meta.py``): every results JSON found by ``results_globs`` must be
compliant, except files grandfathered in ``.integrity/legacy_results.txt``. That list may only shrink.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_KEYS = ("exp_id", "git_commit", "seeds", "config", "command", "timestamp_utc")
MIN_SEEDS = 20


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=20, check=True
        ).stdout.strip()
    except Exception:
        return ""


def build_meta(
    exp_id: str,
    seeds: list[int],
    config: dict[str, Any],
    command: str | None = None,
    tuning_seeds: list[int] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata. ``seeds`` are the held-out TEST seeds."""
    root = repo_root or Path(__file__).resolve().parents[2]
    meta: dict[str, Any] = {
        "exp_id": str(exp_id),
        "git_commit": _git(["rev-parse", "HEAD"], root) or "UNKNOWN",
        "git_dirty": bool(_git(["status", "--porcelain"], root)),
        "seeds": [int(s) for s in seeds],
        "config": config,
        "command": command or " ".join(sys.argv),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
    }
    if tuning_seeds is not None:
        meta["tuning_seeds"] = [int(s) for s in tuning_seeds]
    return meta


def validate_meta(meta: Any, min_seeds: int = MIN_SEEDS) -> list[str]:
    """Return a list of problems (empty list == compliant)."""
    if not isinstance(meta, dict):
        return ["missing top-level '_meta' object"]
    problems = [f"missing key '{k}'" for k in REQUIRED_KEYS if k not in meta]
    seeds = meta.get("seeds")
    if "seeds" in meta:
        if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
            problems.append("'seeds' must be a list of ints")
        elif len(set(seeds)) < min_seeds:
            problems.append(f"only {len(set(seeds))} distinct seeds (< {min_seeds})")
    tuning = meta.get("tuning_seeds")
    if tuning is not None and isinstance(seeds, list) and set(tuning) & set(seeds):
        problems.append("tuning_seeds overlap test seeds (GEMINI.md §4.2)")
    if "git_commit" in meta and meta["git_commit"] in ("", "UNKNOWN"):
        problems.append("git_commit unknown")
    if "config" in meta and not isinstance(meta["config"], dict):
        problems.append("'config' must be an object")
    return problems


def validate_results_file(path: Path, min_seeds: int = MIN_SEEDS) -> list[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # unreadable / not JSON
        return [f"unreadable JSON: {exc}"]
    if not isinstance(data, dict):
        return ["top-level JSON is not an object (cannot hold '_meta')"]
    return validate_meta(data.get("_meta"), min_seeds=min_seeds)


def write_results(path: Path, payload: dict[str, Any], meta: dict[str, Any]) -> Path:
    """Write ``payload`` with ``_meta`` attached; refuses non-compliant metadata."""
    problems = validate_meta(meta)
    if problems:
        raise ValueError("non-compliant results metadata: " + "; ".join(problems))
    out = dict(payload)
    out["_meta"] = meta
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return Path(path)


# ----------------------------------------------------------------------------- gate
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages", "build", "dist",
              ".pytest_cache", ".integrity"}


def find_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == ".integrity" else Path.cwd().resolve()


def _config(root: Path) -> dict:
    p = root / ".integrity" / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def discover_results(root: Path) -> list[str]:
    cfg = _config(root)
    globs = cfg.get("results_globs", ["*result*.json", "**/*result*.json"])
    excludes = cfg.get("results_exclude", [])
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            rel = (Path(dirpath) / fn).relative_to(root).as_posix()
            if not fn.endswith(".json"):
                continue
            if any(fnmatch.fnmatch(rel, g) for g in excludes):
                continue
            if any(fnmatch.fnmatch(rel, g) for g in globs):
                out.append(rel)
    return out


def legacy_path(root: Path) -> Path:
    return root / ".integrity" / "legacy_results.txt"


def load_legacy(root: Path) -> list[str]:
    p = legacy_path(root)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def noncompliant_files(root: Path) -> dict[str, list[str]]:
    bad = {}
    for rel in discover_results(root):
        problems = validate_results_file(root / rel)
        if problems:
            bad[rel] = problems
    return bad


def run_check(root: Path) -> tuple[bool, str]:
    bad = noncompliant_files(root)
    legacy = set(load_legacy(root))
    new_bad = {r: p for r, p in bad.items() if r not in legacy}
    stale = sorted(r for r in legacy if r not in bad)
    msgs = []
    if new_bad:
        msgs.append(f"{len(new_bad)} results file(s) lack compliant run metadata (see GEMINI.md section 6; "
                    f"write them with results_meta.write_results):")
        msgs.extend(f"  {r}: {'; '.join(p)}" for r, p in sorted(new_bad.items()))
    if stale:
        msgs.append("legacy list is stale (file now compliant or deleted) -> remove from "
                    ".integrity/legacy_results.txt: " + ", ".join(stale))
    if msgs:
        return False, "\n".join(msgs)
    return True, f"OK ({len(bad)} legacy non-compliant files grandfathered, 0 new)"


_LEGACY_HEADER = "# Grandfathered results files without run metadata. May only shrink.\n"


def _write_legacy(root: Path, names: list[str]):
    legacy_path(root).parent.mkdir(parents=True, exist_ok=True)
    legacy_path(root).write_text(_LEGACY_HEADER + "".join(n + "\n" for n in sorted(names)), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--init-legacy", action="store_true", help="first-time: grandfather current non-compliant files")
    ap.add_argument("--shrink-legacy", action="store_true", help="drop stale entries (never adds)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    root = (args.root or find_root()).resolve()
    bad = noncompliant_files(root)
    if args.list:
        for r, p in sorted(bad.items()):
            print(f"{r}: {'; '.join(p)}")
        print(f"\n{len(bad)} non-compliant of {len(discover_results(root))} results files")
        return 0
    if args.init_legacy:
        if legacy_path(root).exists():
            print("legacy list exists; refusing to overwrite (it may only shrink).")
            return 2
        _write_legacy(root, list(bad))
        print(f"legacy list written: {len(bad)} files")
        return 0
    if args.shrink_legacy:
        keep = [r for r in load_legacy(root) if r in bad]
        _write_legacy(root, keep)
        print(f"legacy list shrunk to {len(keep)} files")
        return 0
    ok, report = run_check(root)
    print(("results metadata " if ok else "RESULTS METADATA CHECK FAILED: ") + report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
