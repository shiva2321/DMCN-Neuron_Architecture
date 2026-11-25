#!/usr/bin/env python
"""Install the research-integrity kit into ANY project (and optionally globally).

    python install.py <project_dir>                     # install rules + gates into a project
    python install.py <project_dir> --also CLAUDE.md,AGENTS.md   # same rules for other agents
    python install.py --global                          # put the rules in ~/.gemini/GEMINI.md (all projects)
    python install.py <project_dir> --relock            # HUMAN: re-baseline the lock after updating the kit

What it does for a project (idempotent; never overwrites your own data):
  .integrity/            gate tools (lint, results metadata, claims ledger, controls lib, run_gates, templates)
  GEMINI.md              the rules; anything you wrote below the PROJECT-SPECIFIC-NOTES marker is preserved
  docs/CLAIMS.md         claim ledger (created from template if absent)
  .githooks/pre-commit   runs the static gates before every commit (activate: git config core.hooksPath .githooks)
  baselines + lock       existing violations are grandfathered (a ratchet: they can only shrink); rules/gates locked

Nothing is committed and no git config is changed unless you pass --activate-hook.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
MARKER = "<!-- PROJECT-SPECIFIC-NOTES: everything below this line is preserved when the kit is re-installed -->"
CORE_TITLE = "# Research & Engineering Integrity Rules (GEMINI.md)"


def core_text() -> str:
    text = (KIT / "GEMINI.md").read_text(encoding="utf-8")
    return text.split(MARKER, 1)[0].rstrip() + "\n\n" + MARKER + "\n"


def compose_rules(existing: str | None) -> str:
    core = core_text()
    if not existing or not existing.strip():
        return core
    if MARKER in existing:
        appendix = existing.split(MARKER, 1)[1]
        return core + appendix.lstrip("\n")
    return core + "\n## Pre-existing instructions (preserved)\n\n" + existing.strip() + "\n"


def write_if(path: Path, text: str, force: bool, label: str, actions: list[str]):
    if path.exists() and not force:
        actions.append(f"kept existing {label}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    actions.append(f"wrote {label}")


def install_project(target: Path, force: bool, also: list[str], relock: bool, activate_hook: bool, no_init: bool) -> list[str]:
    actions: list[str] = []
    integ = target / ".integrity"
    (integ / "templates").mkdir(parents=True, exist_ok=True)
    for src in sorted((KIT / "tools").glob("*.py")):
        shutil.copy2(src, integ / src.name)
    actions.append("installed tools -> .integrity/*.py")
    for src in sorted((KIT / "templates").iterdir()):
        if src.is_file():
            shutil.copy2(src, integ / "templates" / src.name)
    actions.append("installed templates -> .integrity/templates/")

    write_if(integ / "config.json", (KIT / "templates" / "config.json").read_text(encoding="utf-8"), force, ".integrity/config.json", actions)

    rules_path = target / "GEMINI.md"
    existing = rules_path.read_text(encoding="utf-8") if rules_path.exists() else None
    rules_path.write_text(compose_rules(existing), encoding="utf-8")
    actions.append("wrote GEMINI.md (project notes below the marker preserved)")
    for name in also:
        p = target / name
        old = p.read_text(encoding="utf-8") if p.exists() else None
        p.write_text(compose_rules(old), encoding="utf-8")
        actions.append(f"wrote {name} (same rules)")

    write_if(target / "docs" / "CLAIMS.md", (KIT / "templates" / "CLAIMS.md").read_text(encoding="utf-8"), force, "docs/CLAIMS.md", actions)

    hook = target / ".githooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text((KIT / "templates" / "pre-commit").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    try:
        hook.chmod(0o755)
    except OSError:
        pass
    actions.append("wrote .githooks/pre-commit")

    if activate_hook:
        if (target / ".git").exists():
            subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=str(target), check=False)
            actions.append("activated hook: git config core.hooksPath .githooks")
        else:
            actions.append("not a git repo: hook not activated")
    else:
        actions.append("hook NOT activated - run once: git config core.hooksPath .githooks")

    gates = integ / "run_gates.py"
    if not no_init:
        r = subprocess.run([sys.executable, str(gates), "--root", str(target), "--init"], capture_output=True, text=True)
        actions.extend("  " + ln for ln in (r.stdout + r.stderr).strip().splitlines())
    if relock:
        env = dict(os.environ, INTEGRITY_HUMAN_APPROVED="1")
        r = subprocess.run([sys.executable, str(gates), "--root", str(target), "--update-lock"], capture_output=True, text=True, env=env)
        actions.extend("  " + ln for ln in (r.stdout + r.stderr).strip().splitlines())
    return actions


def install_global(force: bool) -> list[str]:
    path = Path.home() / ".gemini" / "GEMINI.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing.strip() and CORE_TITLE not in existing and MARKER not in existing and not force:
        return [f"{path} already has your own content; refusing to modify it. Re-run with --force to prepend the rules."]
    path.write_text(compose_rules(existing), encoding="utf-8")
    return [f"wrote global rules -> {path} (applies to every project opened with Gemini CLI / Antigravity)"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", type=Path)
    ap.add_argument("--global", dest="glob", action="store_true", help="install rules into ~/.gemini/GEMINI.md")
    ap.add_argument("--also", default="", help="comma list of extra rule files, e.g. CLAUDE.md,AGENTS.md")
    ap.add_argument("--force", action="store_true", help="overwrite config/ledger/own files")
    ap.add_argument("--relock", action="store_true", help="HUMAN: regenerate LOCK.sha256 after updating the kit")
    ap.add_argument("--activate-hook", action="store_true")
    ap.add_argument("--no-init", action="store_true", help="skip baselines/lock creation")
    args = ap.parse_args(argv)

    if not args.target and not args.glob:
        ap.error("give a project directory and/or --global")
    out: list[str] = []
    if args.target:
        t = args.target.resolve()
        if not t.is_dir():
            ap.error(f"{t} is not a directory")
        out.append(f"== project: {t}")
        out += install_project(t, args.force, [s for s in args.also.split(",") if s], args.relock, args.activate_hook, args.no_init)
    if args.glob:
        out.append("== global")
        out += install_global(args.force)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
