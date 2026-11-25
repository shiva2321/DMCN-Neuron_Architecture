#!/usr/bin/env python
"""Run every integrity gate (GEMINI.md §14).  One command, one verdict, exit code 1 on any failure.

    python .integrity/run_gates.py            # lint + results metadata + claims ledger + lock + project tests
    python .integrity/run_gates.py --fast     # static gates only (pre-commit; no project tests)
    python .integrity/run_gates.py --full     # also run config.full_tests and check tracked files stay unmodified
    python .integrity/run_gates.py --only lint,claims
    python .integrity/run_gates.py --init     # first install: create baselines, snapshot, lock (refuses to overwrite)
    INTEGRITY_HUMAN_APPROVED=1 python .integrity/run_gates.py --update-lock    # HUMAN ONLY

Gates
  lint     integrity_lint.py   no NEW fake/curve-fit patterns (baseline may only shrink)
  results  results_meta.py     every results JSON carries run metadata (legacy list may only shrink)
  claims   claims_check.py     ledger valid; levels never raised by agents; status docs match ledger
  lock     LOCK.sha256         gate machinery / rules / integrity tests are unmodified
  tests    config.tests        project-specific null-control tests (see templates/test_domain_controls_example.py)
  tracked  (git)               running the tests did not modify tracked files (evidence must not be overwritten)
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import claims_check  # noqa: E402
import integrity_lint  # noqa: E402
import results_meta  # noqa: E402

DEFAULT_PROTECTED = [
    "GEMINI.md",
    ".integrity/*.py",
    ".integrity/config.json",
    ".integrity/templates/*",
    ".integrity/claim_levels_snapshot.json",
    ".githooks/pre-commit",
]


def find_root() -> Path:
    return HERE.parent if HERE.name == ".integrity" else Path.cwd().resolve()


def load_config(root: Path) -> dict:
    p = root / ".integrity" / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ----------------------------------------------------------------------------- lock
def lock_path(root: Path) -> Path:
    return root / ".integrity" / "LOCK.sha256"


def _sha(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")  # line-ending independent (Windows/autocrlf)
    return hashlib.sha256(data).hexdigest()


def protected_files(root: Path, cfg: dict) -> list[str]:
    patterns = DEFAULT_PROTECTED + list(cfg.get("protected", []))
    found: set[str] = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() and p.name != "LOCK.sha256" and "__pycache__" not in p.parts:
                found.add(p.relative_to(root).as_posix())
    return sorted(found)


def compute_lock(root: Path, cfg: dict) -> dict[str, str]:
    return {rel: _sha(root / rel) for rel in protected_files(root, cfg)}


def write_lock(root: Path, cfg: dict):
    lock = compute_lock(root, cfg)
    lock_path(root).write_text("".join(f"{h}  {rel}\n" for rel, h in sorted(lock.items())), encoding="utf-8")
    return len(lock)


def check_lock(root: Path) -> tuple[bool, str]:
    cfg = load_config(root)
    if not lock_path(root).exists():
        return False, "LOCK.sha256 missing (run: python .integrity/run_gates.py --init)"
    locked = {}
    for ln in lock_path(root).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, rel = ln.split("  ", 1)
            locked[rel.strip()] = h
    current = compute_lock(root, cfg)
    problems = []
    for rel, h in locked.items():
        if rel not in current:
            problems.append(f"protected file missing: {rel}")
        elif current[rel] != h:
            problems.append(f"protected file MODIFIED: {rel}")
    for rel in current:
        if rel not in locked:
            problems.append(f"new protected file not in lock: {rel}")
    if problems:
        return False, "\n".join(problems) + (
            "\nProtected files (rules, gates, integrity tests) may only be changed by the human owner. "
            "Do NOT regenerate the lock yourself; report this to the user.")
    return True, f"OK ({len(locked)} protected files unmodified)"


# ----------------------------------------------------------------------------- tracked files
def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True, timeout=60)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def tracked_modified(root: Path) -> dict[str, str] | None:
    out = _git(root, "status", "--porcelain")
    if out is None:
        return None
    res = {}
    for ln in out.splitlines():
        if ln.startswith("??") or not ln.strip():
            continue
        path = ln[3:].strip().strip('"')
        res[path] = _sha(root / path) if (root / path).is_file() else "DELETED"
    return res


# ----------------------------------------------------------------------------- project tests
def _argv(cmd) -> list[str]:
    """Commands come from the locked config.json. Run as an argument list (never through a shell)."""
    argv = list(cmd) if isinstance(cmd, list) else shlex.split(cmd)
    if argv and argv[0] in ("python", "python3"):
        argv[0] = sys.executable
    return argv


def run_commands(root: Path, cmds: list, timeout: int) -> tuple[bool, str]:
    ok, report = True, []
    for c in cmds:
        cmd, cwd = (c, ".") if isinstance(c, (str, list)) else (c["cmd"], c.get("cwd", "."))
        try:
            r = subprocess.run(_argv(cmd), cwd=str(root / cwd), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            ok = False
            report.append(f"TIMEOUT after {timeout}s: {cmd}")
            continue
        except OSError as exc:
            ok = False
            report.append(f"could not run {cmd}: {exc}")
            continue
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-8:])
        report.append(f"$ {cmd}  (cwd={cwd})  exit={r.returncode}\n{tail}")
        ok = ok and r.returncode == 0
    return ok, "\n".join(report)


# ----------------------------------------------------------------------------- init
def init_all(root: Path) -> list[str]:
    cfg = load_config(root)
    msgs = []
    if not integrity_lint.baseline_path(root).exists():
        cur = integrity_lint.collect(root)
        integrity_lint.write_baseline(root, cur)
        msgs.append(f"lint baseline: {len(cur)} grandfathered violations")
    if not results_meta.legacy_path(root).exists():
        bad = results_meta.noncompliant_files(root)
        results_meta._write_legacy(root, list(bad))
        msgs.append(f"legacy results list: {len(bad)} files")
    ledger = root / cfg.get("ledger", "docs/CLAIMS.md")
    if ledger.exists() and not claims_check.snapshot_path(root).exists():
        claims_check.write_snapshot(root, claims_check.parse_ledger(ledger))
        msgs.append("claim level snapshot written")
    if not lock_path(root).exists():
        msgs.append(f"lock written: {write_lock(root, cfg)} protected files")
    return msgs or ["nothing to initialise (all baselines already exist)"]


# ----------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--fast", action="store_true", help="static gates only")
    ap.add_argument("--full", action="store_true", help="also run config.full_tests + tracked-file check")
    ap.add_argument("--only", default="", help="comma list of: lint,results,claims,lock,tests,tracked")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--update-lock", action="store_true")
    args = ap.parse_args(argv)
    root = (args.root or find_root()).resolve()
    cfg = load_config(root)

    if args.init:
        for m in init_all(root):
            print("init:", m)
        return 0
    if args.update_lock:
        if os.environ.get("INTEGRITY_HUMAN_APPROVED") != "1":
            print("refusing: updating the lock is a human action. Set INTEGRITY_HUMAN_APPROVED=1 if you are the owner.")
            return 2
        print(f"lock updated: {write_lock(root, cfg)} protected files")
        return 0

    wanted = {s for s in args.only.split(",") if s} or {"lint", "results", "claims", "lock"} | (
        set() if args.fast else {"tests"}) | ({"tracked"} if args.full else set())
    results: list[tuple[str, bool, str]] = []

    if "lint" in wanted:
        results.append(("lint", *integrity_lint.run_check(root)))
    if "results" in wanted:
        results.append(("results", *results_meta.run_check(root)))
    if "claims" in wanted:
        results.append(("claims", *claims_check.run_check(root)))
    if "lock" in wanted:
        results.append(("lock", *check_lock(root)))

    timeout = int(cfg.get("tests_timeout", 1800))
    before = tracked_modified(root) if ("tracked" in wanted or args.full) else None
    if "tests" in wanted and cfg.get("tests"):
        results.append(("tests", *run_commands(root, cfg["tests"], timeout)))
    elif "tests" in wanted:
        results.append(("tests", True, "no project tests configured (config.tests is empty) - "
                                       "add domain null-control tests (templates/test_domain_controls_example.py)"))
    if args.full and cfg.get("full_tests"):
        results.append(("full_tests", *run_commands(root, cfg["full_tests"], timeout)))
    if before is not None:
        after = tracked_modified(root)
        if after is None:
            results.append(("tracked", True, "not a git repository; skipped"))
        else:
            allow = cfg.get("tracked_write_allowlist", [])
            changed = [p for p, h in after.items() if before.get(p) != h
                       and not any(fnmatch.fnmatch(p, a) for a in allow)]
            if changed:
                results.append(("tracked", False,
                                "tests/benchmarks modified tracked files (evidence must not be overwritten, "
                                "GEMINI.md section 6): " + ", ".join(sorted(changed))))
            else:
                results.append(("tracked", True, "OK (no tracked file changed by the tests)"))

    failed = False
    for name, ok, report in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: " + report.replace("\n", "\n         "))
        failed = failed or not ok
    print("\nINTEGRITY GATES: " + ("FAILED" if failed else "all passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
