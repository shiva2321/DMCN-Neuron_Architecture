#!/usr/bin/env python
"""Static integrity lint for ANY Python project (GEMINI.md §2, §8, §14).

Flags patterns used to fake or inflate results:

  LITERAL_RESULT_STRING          result numbers / competitor numbers typed into print strings
  LITERAL_STATUS_IN_LOG          'True' / 'False' typed into a log line instead of measured
  FORCED_SUCCESS_FLAG            success flag forced True by a "last iteration" condition
  LABEL_CONDITIONED_STIMULUS     stimulus (phase/input) chosen by comparing the label/pattern
  PASS_REQUIRES_BASELINE_FAILURE pass criterion that needs a baseline to score low
  BASELINE_RELATIVE_MARGIN       pass criterion defined as a margin over a baseline
  SWALLOWED_EXCEPTION            `except ...: pass` (silent fallback hides failure)
  UNUSED_LOCAL_VARIABLE          computed and never read (decorative computation)
  UNUSED_STATE_NEVER_READ        self.<attr> assigned but never read anywhere in the code base

Existing violations are grandfathered in ``.integrity/lint_baseline.json`` (a ratchet): the lint FAILS
on any *new* violation and on any *stale* baseline entry (fixed -> baseline must shrink).
Justified false positives need an inline comment on the same or previous line::

    # integrity-lint: allow RULE_NAME <specific reason, at least 10 characters>

Usage (from anywhere):
    python .integrity/integrity_lint.py                  # check vs baseline
    python .integrity/integrity_lint.py --list           # print every current violation
    python .integrity/integrity_lint.py --init-baseline  # first-time baseline (refuses to overwrite)
    python .integrity/integrity_lint.py --shrink-baseline  # drop fixed entries; never adds new ones
    python .integrity/integrity_lint.py --root PATH
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env", "site-packages",
    "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", ".integrity", ".ipynb_checkpoints",
}

ALLOW_RE = re.compile(r"#\s*integrity-lint:\s*allow\s+([A-Z_]+)\s+(\S.{9,})")
SUCCESS_NAME_RE = re.compile(
    r"^(reached_goal|goal_reached|success|succeeded|solved|passed|correct|all_correct|is_correct|ok)$", re.I
)
LABEL_NAMES = {
    "pat", "pat_str", "pattern", "pattern_name", "label", "labels", "target", "y",
    "cls", "class_id", "expected", "truth", "answer",
}
STIMULUS_HINT_RE = re.compile(r"(phase|signal|sig\d*|inp|input|stim|drive)", re.I)
BASELINE_NAME_RE = re.compile(
    r"(single|baseline|monolith|\blif\b|^lif|mlp|perceptron|tdl|memoryless|tapped|control|naive)", re.I
)

_PCT = re.compile(r"\d+(?:\.\d+)?\s*%")
_TIMES = re.compile(r"\b\d+(?:\.\d+)?x\b")
_MODEL = re.compile(r"\b(?:RCPN|MLP|LIF|TDL|Perc|Memoryless|VSA|HPR|SOTA|Baseline|None)\b\s*:\s*[-+]?\d")
_DEC = re.compile(r"(?<![=<>\w.\-])\d+\.\d+")
_STATUS = re.compile(r"\b(True|False)\b")


@dataclass(frozen=True)
class Violation:
    rule: str
    file: str
    line: int
    snippet: str
    sig: str

    def render(self) -> str:
        return f"{self.file}:{self.line}: [{self.rule}] {self.snippet}"


def _sig(rule: str, file: str, snippet: str) -> str:
    norm = re.sub(r"\s+", " ", snippet.strip())
    return hashlib.sha1(f"{rule}|{file}|{norm}".encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- AST helpers
def _iter_print_strings(node: ast.AST):
    """Yield string Constant nodes inside an expression, skipping f-string format specs."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node
    elif isinstance(node, ast.JoinedStr):
        for v in node.values:
            yield from _iter_print_strings(v)
    elif isinstance(node, ast.FormattedValue):
        yield from _iter_print_strings(node.value)  # format_spec deliberately skipped
    else:
        for child in ast.iter_child_nodes(node):
            yield from _iter_print_strings(child)


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    return None


def _is_last_iteration_compare(cmp_: ast.Compare) -> bool:
    parts = [cmp_.left, *cmp_.comparators]
    return any(
        isinstance(p, ast.BinOp) and isinstance(p.op, ast.Sub)
        and isinstance(p.right, ast.Constant) and p.right.value == 1
        for p in parts
    )


def _contains_last_iteration_or(test: ast.AST) -> bool:
    for n in ast.walk(test):
        if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.Or):
            if any(isinstance(v, ast.Compare) and _is_last_iteration_compare(v) for v in n.values):
                return True
    return False


def _subtree_mentions_stimulus(nodes: list[ast.AST]) -> bool:
    for root in nodes:
        for n in ast.walk(root):
            if isinstance(n, ast.Name) and (n.id == "pi" or STIMULUS_HINT_RE.search(n.id)):
                return True
            if isinstance(n, ast.Attribute) and (n.attr == "pi" or STIMULUS_HINT_RE.search(n.attr)):
                return True
    return False


def _label_compare(test: ast.AST) -> bool:
    for n in ast.walk(test):
        if isinstance(n, ast.Compare) and isinstance(n.ops[0], (ast.Eq, ast.In, ast.Is)):
            if _name_of(n.left) in LABEL_NAMES and isinstance(n.comparators[0], (ast.Constant, ast.Tuple, ast.List)):
                return True
    return False


# ----------------------------------------------------------------------------- per-file rules
def lint_source(rel: str, source: str) -> list[Violation]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # pre-existing code may contain sloppy string escapes; not our concern
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines()
    found: list[tuple[str, int]] = []

    def add(rule: str, lineno: int):
        found.append((rule, lineno))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            for arg in node.args:
                for c in _iter_print_strings(arg):
                    s = c.value
                    if _PCT.search(s) or _TIMES.search(s) or _MODEL.search(s) or _DEC.search(s):
                        add("LITERAL_RESULT_STRING", c.lineno)
                    if _STATUS.search(s):
                        add("LITERAL_STATUS_IN_LOG", c.lineno)

        if isinstance(node, ast.If) and _contains_last_iteration_or(node.test):
            for stmt in node.body:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value is True:
                        if any(_name_of(t) and SUCCESS_NAME_RE.match(_name_of(t) or "") for t in n.targets):
                            add("FORCED_SUCCESS_FLAG", n.lineno)

        if isinstance(node, ast.IfExp) and _label_compare(node.test):
            if _subtree_mentions_stimulus([node.body, node.orelse]):
                add("LABEL_CONDITIONED_STIMULUS", node.lineno)
        if isinstance(node, ast.Assign) and any(STIMULUS_HINT_RE.search(_name_of(t) or "") for t in node.targets):
            # e.g.  sig = f(t, 0.0 if label == "a" else 3.14): a stimulus built by branching on the label
            if any(isinstance(n, ast.IfExp) and _label_compare(n.test) for n in ast.walk(node.value)):
                add("LABEL_CONDITIONED_STIMULUS", node.lineno)
        if isinstance(node, ast.If) and _label_compare(node.test):
            for stmt in [*node.body, *node.orelse]:
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Assign) and any(
                        STIMULUS_HINT_RE.search(_name_of(t) or "") for t in n.targets
                    ):
                        add("LABEL_CONDITIONED_STIMULUS", node.lineno)

        if isinstance(node, ast.Compare):
            left_name = _name_of(node.left)
            op = node.ops[0]
            if left_name and BASELINE_NAME_RE.search(left_name) and isinstance(op, (ast.Lt, ast.LtE)):
                if isinstance(node.comparators[0], (ast.Constant, ast.Attribute, ast.Name)):
                    add("PASS_REQUIRES_BASELINE_FAILURE", node.lineno)
            if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Sub) and isinstance(op, (ast.Gt, ast.GtE)):
                names = [_name_of(node.left.left), _name_of(node.left.right)]
                if any(n and BASELINE_NAME_RE.search(n) for n in names):
                    add("BASELINE_RELATIVE_MARGIN", node.lineno)

        if isinstance(node, ast.ExceptHandler):
            if all(isinstance(b, ast.Pass) for b in node.body):
                add("SWALLOWED_EXCEPTION", node.lineno)

    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        stored: dict[str, int] = {}
        loaded: set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                loaded.add(n.id)
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                stored.setdefault(n.targets[0].id, n.lineno)
        for name, ln in stored.items():
            if name not in loaded and not name.startswith("_"):
                add("UNUSED_LOCAL_VARIABLE", ln)

    out: list[Violation] = []
    seen: set[tuple[str, int]] = set()
    for rule, lineno in found:
        if (rule, lineno) in seen:
            continue
        seen.add((rule, lineno))
        snippet = lines[lineno - 1].strip()[:160] if 0 < lineno <= len(lines) else ""
        window = " ".join(lines[max(0, lineno - 2):lineno])
        m = ALLOW_RE.search(window)
        if m and m.group(1) == rule:
            continue
        out.append(Violation(rule, rel, lineno, snippet, _sig(rule, rel, snippet)))
    return out


# ----------------------------------------------------------------------------- project scan
def find_root(start: Path | None = None) -> Path:
    """Project root = parent of the .integrity dir this file lives in; else cwd."""
    here = Path(__file__).resolve().parent
    if here.name == ".integrity":
        return here.parent
    return (start or Path.cwd()).resolve()


def load_config(root: Path) -> dict:
    p = root / ".integrity" / "config.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def iter_files(root: Path, cfg: dict | None = None):
    cfg = cfg if cfg is not None else load_config(root)
    excludes = list(cfg.get("lint_exclude", []))
    skip_dirs = DEFAULT_SKIP_DIRS | set(cfg.get("lint_skip_dirs", []))
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            p = Path(dirpath) / fn
            rel = p.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
                continue
            yield p


def unused_state_violations(root: Path, cfg: dict) -> list[Violation]:
    """self.<attr> stored but never loaded anywhere in the scanned code base."""
    stores: dict[str, list[tuple[str, int]]] = {}
    loads: set[str] = set()
    sources: dict[str, list[str]] = {}
    for p in iter_files(root, cfg):
        rel = p.relative_to(root).as_posix()
        try:
            src = p.read_text(encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(src)
        except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
            continue
        sources[rel] = src.splitlines()
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                if isinstance(n.ctx, ast.Load):
                    loads.add(n.attr)
                elif isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) and n.value.id == "self":
                    stores.setdefault(n.attr, []).append((rel, n.lineno))
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                loads.add(n.value)  # getattr(obj, "name") / dict-style access
    out = []
    for attr, locs in stores.items():
        if attr in loads or attr.startswith("_"):
            continue
        for rel, ln in locs:
            text = sources[rel][ln - 1].strip()[:160]
            window = " ".join(sources[rel][max(0, ln - 2):ln])
            m = ALLOW_RE.search(window)
            if m and m.group(1) == "UNUSED_STATE_NEVER_READ":
                continue
            out.append(Violation("UNUSED_STATE_NEVER_READ", rel, ln, text, _sig("UNUSED_STATE_NEVER_READ", rel, text)))
    return out


def collect(root: Path) -> list[Violation]:
    cfg = load_config(root)
    found: list[Violation] = []
    for p in iter_files(root, cfg):
        rel = p.relative_to(root).as_posix()
        try:
            found.extend(lint_source(rel, p.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    found.extend(unused_state_violations(root, cfg))
    return sorted(found, key=lambda v: (v.file, v.line, v.rule))


def baseline_path(root: Path) -> Path:
    return root / ".integrity" / "lint_baseline.json"


def load_baseline(root: Path) -> Counter:
    path = baseline_path(root)
    if not path.exists():
        return Counter()
    data = json.loads(path.read_text(encoding="utf-8"))
    return Counter({e["sig"]: int(e.get("count", 1)) for e in data["violations"]})


def diff_against_baseline(current: list[Violation], baseline: Counter):
    cur = Counter(v.sig for v in current)
    budget = Counter(cur - baseline)
    stale = baseline - cur
    new = []
    for v in current:
        if budget[v.sig] > 0:
            new.append(v)
            budget[v.sig] -= 1
    return new, stale


def write_baseline(root: Path, violations: list[Violation]):
    grouped: dict[str, dict] = {}
    for v in violations:
        e = grouped.setdefault(v.sig, {**asdict(v), "count": 0})
        e["count"] += 1
        e.pop("line", None)
    payload = {
        "version": 1,
        "note": "Grandfathered violations. May only shrink. Fix the code, then run --shrink-baseline.",
        "violations": sorted(grouped.values(), key=lambda e: (e["file"], e["rule"], e["snippet"])),
    }
    path = baseline_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_check(root: Path) -> tuple[bool, str]:
    current = collect(root)
    baseline = load_baseline(root)
    new, stale = diff_against_baseline(current, baseline)
    msgs = []
    if new:
        msgs.append(f"{len(new)} NEW violation(s) (fix them, or add a justified "
                    f"'# integrity-lint: allow RULE reason' comment):")
        msgs.extend("  " + v.render() for v in new)
    if stale:
        msgs.append(f"{sum(stale.values())} baseline entr(ies) no longer occur -> run "
                    f"'python .integrity/integrity_lint.py --shrink-baseline' so the baseline only shrinks.")
    if not msgs:
        return True, f"OK ({len(current)} grandfathered, 0 new)"
    return False, "\n".join(msgs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--list", action="store_true", help="print all current violations")
    ap.add_argument("--init-baseline", action="store_true")
    ap.add_argument("--shrink-baseline", action="store_true")
    args = ap.parse_args(argv)
    root = (args.root or find_root()).resolve()

    current = collect(root)
    if args.list:
        for v in current:
            print(v.render())
        print(f"\n{len(current)} violations", dict(Counter(v.rule for v in current)))
        return 0
    if args.init_baseline:
        if baseline_path(root).exists():
            print("baseline exists; refusing to overwrite (it may only shrink). Use --shrink-baseline.")
            return 2
        write_baseline(root, current)
        print(f"baseline written: {len(current)} grandfathered violations")
        return 0
    if args.shrink_baseline:
        baseline = load_baseline(root)
        _, stale = diff_against_baseline(current, baseline)
        remaining = Counter(Counter(v.sig for v in current) & baseline)
        kept = []
        for v in current:
            if remaining[v.sig] > 0:
                kept.append(v)
                remaining[v.sig] -= 1
        write_baseline(root, kept)
        print(f"baseline shrunk: removed {sum(stale.values())} fixed entries; {len(kept)} remain")
        return 0

    ok, report = run_check(root)
    print(("integrity lint " if ok else "INTEGRITY LINT FAILED: ") + report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
