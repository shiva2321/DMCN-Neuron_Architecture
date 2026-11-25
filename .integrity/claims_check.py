#!/usr/bin/env python
"""Claim-ledger gate (GEMINI.md §9, §10) — works for any project.

Checks, in order:
  1. The ledger (default ``docs/CLAIMS.md``) parses and every row has a valid level.
  2. Rows at REPRODUCED or above name a non-empty gate (tests that enforce them).
  3. Levels never RISE relative to ``.integrity/claim_levels_snapshot.json`` (lowering is always
     allowed; raising or adding a row above OBSERVED requires a human to re-baseline the snapshot).
  4. Status documents listed in ``config.status_docs`` (README / BOARD / ROADMAP ...): every table
     row whose first cell is a ledger ID must carry a ``[LEVEL]`` tag equal to the ledger level, and
     must not claim a stronger level.  Documents in ``config.status_docs_pending`` are reported as
     warnings while inconsistent and FAIL once consistent (so the pending list only shrinks).

Ledger row format (markdown table):  | ID | Level | Claim | Evidence | Gate |
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

RANK = {"REFUTED": 0, "UNAUDITED": 0, "IMPLEMENTED": 1, "OBSERVED": 2,
        "REPRODUCED": 3, "CONTROLLED": 4, "GENERALIZES": 5}
NEW_ROW_MAX_RANK = RANK["OBSERVED"]
LEVEL_TAG_RE = re.compile(r"\[(REFUTED|UNAUDITED|IMPLEMENTED|OBSERVED|REPRODUCED|CONTROLLED|GENERALIZES)\]")


def find_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == ".integrity" else Path.cwd().resolve()


def load_config(root: Path) -> dict:
    p = root / ".integrity" / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _norm_id(cell: str) -> str:
    return re.sub(r"[\*`]", "", cell).strip()


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_ledger(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.lstrip().startswith("|"):
            continue
        cells = _cells(ln)
        if len(cells) < 3 or set("".join(cells)) <= set("-: "):
            continue
        if cells[0].lower() in ("id",):
            continue
        rows.append({"id": _norm_id(cells[0]), "level": _norm_id(cells[1]).upper(),
                     "claim": cells[2] if len(cells) > 2 else "",
                     "evidence": cells[3] if len(cells) > 3 else "",
                     "gate": cells[4] if len(cells) > 4 else ""})
    return rows


def snapshot_path(root: Path) -> Path:
    return root / ".integrity" / "claim_levels_snapshot.json"


def write_snapshot(root: Path, rows: list[dict]):
    snapshot_path(root).write_text(json.dumps({r["id"]: r["level"] for r in rows}, indent=2), encoding="utf-8")


def status_doc_problems(root: Path, doc: str, ledger: dict[str, str], id_re: re.Pattern) -> list[str]:
    problems = []
    for i, ln in enumerate((root / doc).read_text(encoding="utf-8").splitlines(), 1):
        if not ln.lstrip().startswith("|"):
            continue
        cells = _cells(ln)
        if not cells:
            continue
        cid = _norm_id(cells[0])
        if not id_re.match(cid):
            continue
        if cid not in ledger:
            problems.append(f"{doc}:{i}: '{cid}' has no row in the claim ledger")
            continue
        tags = LEVEL_TAG_RE.findall(ln)
        if not tags:
            problems.append(f"{doc}:{i}: '{cid}' missing [LEVEL] tag (ledger says {ledger[cid]})")
        elif RANK[tags[0]] > RANK[ledger[cid]] or tags[0] != ledger[cid]:
            problems.append(f"{doc}:{i}: '{cid}' tagged [{tags[0]}] but ledger says {ledger[cid]}")
    return problems


_SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages", "build", "dist", ".pytest_cache"}
GATE_TOKEN_RE = re.compile(r"(test_[A-Za-z0-9_]+(?:\.py)?)")


def known_test_names(root: Path) -> set[str]:
    """File names and `def test_*` names anywhere in the project (so ledger gates cannot point at nothing)."""
    names: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP and not d.endswith(".egg-info")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            names.add(fn)
            try:
                text = (Path(dirpath) / fn).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            names.update(re.findall(r"def (test_\w+)", text))
    return names


def missing_gate_refs(root: Path, rows: list[dict]) -> list[str]:
    known = known_test_names(root)
    out = []
    for r in rows:
        for tok in GATE_TOKEN_RE.findall(r["gate"]):
            if tok not in known:
                out.append(f"{r['id']}: gate '{tok}' does not exist in the project")
    return out


def run_check(root: Path) -> tuple[bool, str]:
    cfg = load_config(root)
    ledger_rel = cfg.get("ledger", "docs/CLAIMS.md")
    ledger_path = root / ledger_rel
    if not ledger_path.exists():
        return False, f"claim ledger missing: {ledger_rel} (create it from .integrity/templates/CLAIMS.md)"
    rows = parse_ledger(ledger_path)
    problems: list[str] = []
    warnings: list[str] = []
    if not rows:
        warnings.append("ledger has no claim rows yet (fine for a new project; add one per claim you make)")
    ids = [r["id"] for r in rows]
    problems += [f"duplicate ledger ID '{i}'" for i in sorted({x for x in ids if ids.count(x) > 1})]
    for r in rows:
        if r["level"] not in RANK:
            problems.append(f"{r['id']}: invalid level '{r['level']}'")
            continue
        if RANK[r["level"]] >= RANK["REPRODUCED"] and (not r["gate"] or r["gate"].lower() in ("none", "-", "n/a")):
            problems.append(f"{r['id']}: level {r['level']} requires a named gate (tests that enforce it)")
        if r["level"] == "REFUTED" and not r["evidence"]:
            problems.append(f"{r['id']}: REFUTED requires evidence")
    problems += missing_gate_refs(root, rows)
    levels = {r["id"]: r["level"] for r in rows if r["level"] in RANK}

    snap_p = snapshot_path(root)
    if snap_p.exists():
        snap = json.loads(snap_p.read_text(encoding="utf-8"))
        for cid, lvl in levels.items():
            if cid in snap:
                if RANK.get(snap[cid], 0) < RANK[lvl]:
                    problems.append(f"{cid}: level raised {snap[cid]} -> {lvl} without human re-baseline "
                                    f"(snapshot is locked; claims may only be lowered by agents)")
            elif RANK[lvl] > NEW_ROW_MAX_RANK:
                problems.append(f"{cid}: new row at {lvl}; new rows may start at most at OBSERVED")
    else:
        warnings.append("no claim_levels_snapshot.json (run_gates.py --init creates it)")

    id_re = re.compile(cfg.get("status_id_regex", r"^[A-Za-z]+-\d+"))
    pending = set(cfg.get("status_docs_pending", []))
    for doc in cfg.get("status_docs", []):
        if not (root / doc).exists():
            problems.append(f"status doc missing: {doc}")
            continue
        dp = status_doc_problems(root, doc, levels, id_re)
        if doc in pending:
            if dp:
                warnings.append(f"{doc}: {len(dp)} rows not yet reconciled with the ledger (pending): e.g. {dp[0]}")
            else:
                problems.append(f"{doc} is consistent with the ledger -> remove it from status_docs_pending")
        else:
            problems += dp

    if problems:
        return False, "\n".join([*problems, *[f"warning: {w}" for w in warnings]])
    msg = f"OK ({len(rows)} claims)"
    if warnings:
        msg += "\n" + "\n".join(f"  warning: {w}" for w in warnings)
    return True, msg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--snapshot", action="store_true", help="HUMAN ONLY: re-baseline claim levels")
    args = ap.parse_args(argv)
    root = (args.root or find_root()).resolve()
    if args.snapshot:
        if os.environ.get("INTEGRITY_HUMAN_APPROVED") != "1":
            print("refusing: re-baselining claim levels is a human action (set INTEGRITY_HUMAN_APPROVED=1)")
            return 2
        cfg = load_config(root)
        write_snapshot(root, parse_ledger(root / cfg.get("ledger", "docs/CLAIMS.md")))
        print("claim level snapshot written")
        return 0
    ok, report = run_check(root)
    print(("claims ledger " if ok else "CLAIMS LEDGER CHECK FAILED:\n") + report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
