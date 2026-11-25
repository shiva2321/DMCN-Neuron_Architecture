"""Self-tests for the research-integrity kit. Run:  python -m pytest research-integrity-kit/tests -q

These prove the gates can both PASS honest work and FAIL on each kind of cheating, and that the installer
produces a working, tamper-evident setup in an arbitrary project.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / "tools"))

import claims_check  # noqa: E402
import controls  # noqa: E402
import integrity_lint as lint  # noqa: E402
import results_meta  # noqa: E402


def rules(src: str, rel: str = "x.py") -> set[str]:
    return {v.rule for v in lint.lint_source(rel, textwrap.dedent(src))}


# ----------------------------------------------------------------------------- lint: each rule fires
@pytest.mark.parametrize("src,rule", [
    ('print("RCPN: 98.8%")', "LITERAL_RESULT_STRING"),
    ('print(f"{x:<16} {\'MLP:  78.8%\':<16}")', "LITERAL_RESULT_STRING"),
    ('print("Decays to 0.44 without Dict")', "LITERAL_RESULT_STRING"),
    ('print("speedup 12x")', "LITERAL_RESULT_STRING"),
    ('print(f"{ep} | True | 0")', "LITERAL_STATUS_IN_LOG"),
    ("for step in range(n):\n    if check(pos) or step == n - 1:\n        reached_goal = True\n", "FORCED_SUCCESS_FLAG"),
    ("phase1 = np.pi if pat == (1, 1) else 0.0", "LABEL_CONDITIONED_STIMULUS"),
    ('sig = f(t, 0.0 if label == "a" else 3.14)', "LABEL_CONDITIONED_STIMULUS"),
    ("if pattern == (1, 1):\n    phase = 3.14\n", "LABEL_CONDITIONED_STIMULUS"),
    ("cap_pass = rcc_acc >= 0.95 and single_acc <= 0.60", "PASS_REQUIRES_BASELINE_FAILURE"),
    ("ok = ours - baseline_acc >= 0.4", "BASELINE_RELATIVE_MARGIN"),
    ("try:\n    run()\nexcept Exception:\n    pass\n", "SWALLOWED_EXCEPTION"),
    ("def f():\n    c_norm = compute()\n    return 1\n", "UNUSED_LOCAL_VARIABLE"),
])
def test_lint_rule_fires(src, rule):
    assert rule in rules(src)


@pytest.mark.parametrize("src", [
    'print(f"accuracy {acc:.1f}%")',                      # measured value via format spec, no literal number
    'print(f"MSE {mse:>10.6f}")',
    'print("threshold (<-0.75)")',                        # threshold context, not a result
    'print("tau=20.0")',
    "phase1 = np.pi if use_antiphase else 0.0",           # not conditioned on a label
    "ok = ours >= 0.95",
    "def f():\n    c = g()\n    return c\n",
    "try:\n    run()\nexcept ValueError as e:\n    log(e)\n",
])
def test_lint_clean_code_is_clean(src):
    assert rules(src) == set()


def test_lint_allow_comment_requires_specific_reason():
    ok = 'print("RCPN: 98.8%")  # integrity-lint: allow LITERAL_RESULT_STRING quoted from cited paper table 3\n'
    weak = 'print("RCPN: 98.8%")  # integrity-lint: allow LITERAL_RESULT_STRING ok\n'
    assert "LITERAL_RESULT_STRING" not in rules(ok)
    assert "LITERAL_RESULT_STRING" in rules(weak)


# ----------------------------------------------------------------------------- controls library
def test_wilson_and_rate():
    r = controls.rate([True] * 3)
    assert r.p == 1.0 and r.ci[0] < 0.5  # "3 of 3" is NOT "100%"
    assert controls.rate([True] * 20).ci[0] > 0.8


def test_gates_pass_honest_and_fail_cheaters():
    good, bad_ctrl = controls.Rate(18, 20), controls.Rate(17, 20)
    assert controls.gate_beats_controls(good, {"heuristic": controls.Rate(2, 20)}) == []
    assert controls.gate_beats_controls(good, {"heuristic": bad_ctrl})          # control matches -> failure
    assert controls.gate_depends_on_inputs(controls.Rate(18, 20), controls.Rate(2, 20)) == []
    assert controls.gate_depends_on_inputs(controls.Rate(18, 20), controls.Rate(16, 20))
    assert controls.gate_absent_in_null(0.0, 0.5) == [] and controls.gate_absent_in_null(1.0, 0.5)
    assert controls.gate_metric_improved(0.669, 0.670, higher_is_better=False)   # RMSE flat -> failure
    assert controls.gate_metric_improved(0.669, 0.40, higher_is_better=False) == []
    assert controls.gate_min_seeds(3) and controls.gate_min_seeds(20) == []
    with pytest.raises(AssertionError):
        controls.assert_gate(["x"])


def test_summarize_and_permutation():
    s = controls.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["mean"] == 3.0 and s["min"] == 1.0 and s["max"] == 5.0 and s["ci95"][0] < 3.0 < s["ci95"][1]
    acc = lambda y, p: sum(a == b for a, b in zip(y, p)) / len(y)  # noqa: E731
    y = [0, 1] * 30
    assert controls.permutation_pvalue(acc, y, y, n_perm=200) < 0.05
    rnd = [0, 0, 1, 1, 1, 0] * 10
    assert controls.permutation_pvalue(acc, y, rnd, n_perm=200) > 0.05


def test_template_example_passes_in_kit():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(KIT / "templates" / "test_domain_controls_example.py")],
                       capture_output=True, text=True, cwd=str(KIT))
    assert r.returncode == 0, r.stdout + r.stderr


# ----------------------------------------------------------------------------- results metadata
def good_meta(n=20):
    return {"exp_id": "e1", "git_commit": "abc", "seeds": list(range(n)), "config": {}, "command": "x",
            "timestamp_utc": "t"}


def test_meta_validation():
    assert results_meta.validate_meta(good_meta()) == []
    assert results_meta.validate_meta(good_meta(5))                      # too few seeds
    assert results_meta.validate_meta(None)
    m = good_meta()
    m["tuning_seeds"] = [0, 1]
    assert results_meta.validate_meta(m)                                 # tuning overlaps test seeds
    with pytest.raises(ValueError):
        results_meta.write_results(Path("/nonexistent/x.json"), {}, good_meta(3))


# ----------------------------------------------------------------------------- end-to-end install
def run(args, cwd, env=None, check=False):
    e = dict(os.environ)
    e.pop("INTEGRITY_HUMAN_APPROVED", None)
    if env:
        e.update(env)
    return subprocess.run([sys.executable, *args], cwd=str(cwd), capture_output=True, text=True, env=e)


@pytest.fixture()
def project(tmp_path):
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "old_cheat.py").write_text(
        'def score():\n    print("RCPN: 98.8%")\n    return 1\n', encoding="utf-8")        # pre-existing violation
    (tmp_path / "bench" / "old_results.json").write_text('{"acc": 1.0}', encoding="utf-8")  # pre-existing legacy file
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=False)
    r = run([str(KIT / "install.py"), str(tmp_path)], KIT)
    assert r.returncode == 0, r.stdout + r.stderr
    return tmp_path


def gates(project, *extra, env=None):
    return run([".integrity/run_gates.py", *extra], project, env=env)


def test_install_creates_working_tamper_evident_setup(project):
    for rel in (".integrity/run_gates.py", ".integrity/LOCK.sha256", ".integrity/lint_baseline.json",
                ".integrity/legacy_results.txt", "GEMINI.md", "docs/CLAIMS.md", ".githooks/pre-commit"):
        assert (project / rel).exists(), rel
    r = gates(project, "--fast")
    assert r.returncode == 0, r.stdout + r.stderr                  # existing mess is grandfathered
    assert "all passed" in r.stdout


def test_new_fake_result_string_is_blocked(project):
    (project / "bench" / "new.py").write_text('print("MLP:  78.8%")\n', encoding="utf-8")
    r = gates(project, "--fast")
    assert r.returncode == 1 and "LITERAL_RESULT_STRING" in r.stdout and "new.py" in r.stdout


def test_new_label_conditioned_stimulus_is_blocked(project):
    (project / "bench" / "xor.py").write_text("phase1 = np.pi if pat == (1, 1) else 0.0\n", encoding="utf-8")
    assert "LABEL_CONDITIONED_STIMULUS" in gates(project, "--fast").stdout


def test_fixing_old_violation_forces_baseline_shrink(project):
    (project / "bench" / "old_cheat.py").write_text("def score():\n    return 1\n", encoding="utf-8")
    r = gates(project, "--only", "lint")
    assert r.returncode == 1 and "shrink-baseline" in r.stdout
    assert run([".integrity/integrity_lint.py", "--shrink-baseline"], project).returncode == 0
    assert gates(project, "--only", "lint").returncode == 0


def test_new_results_file_needs_metadata_but_compliant_one_passes(project):
    (project / "results").mkdir()
    (project / "results" / "new_result.json").write_text('{"acc": 1.0}', encoding="utf-8")
    r = gates(project, "--only", "results")
    assert r.returncode == 1 and "new_result.json" in r.stdout
    results_meta.write_results(project / "results" / "new_result.json", {"acc": 0.8}, good_meta())
    assert gates(project, "--only", "results").returncode == 0


def test_rules_and_gates_are_tamper_evident_and_lock_is_human_only(project):
    assert gates(project, "--only", "lock").returncode == 0
    (project / "GEMINI.md").write_text((project / "GEMINI.md").read_text(encoding="utf-8") + "\nweakened\n", encoding="utf-8")
    r = gates(project, "--only", "lock")
    assert r.returncode == 1 and "MODIFIED: GEMINI.md" in r.stdout
    assert gates(project, "--update-lock").returncode == 2           # agent without human approval cannot relock
    assert gates(project, "--update-lock", env={"INTEGRITY_HUMAN_APPROVED": "1"}).returncode == 0
    assert gates(project, "--only", "lock").returncode == 0
    (project / ".integrity" / "integrity_lint.py").write_text("# gutted\n", encoding="utf-8")
    assert gates(project, "--only", "lock").returncode == 1


def test_line_endings_do_not_break_lock(project):
    p = project / "GEMINI.md"
    lf = p.read_bytes().replace(b"\r\n", b"\n")
    p.write_bytes(lf.replace(b"\n", b"\r\n"))          # CRLF checkout (Windows / autocrlf)
    assert gates(project, "--only", "lock").returncode == 0
    p.write_bytes(lf)                                   # LF checkout
    assert gates(project, "--only", "lock").returncode == 0


def write_ledger(project, rows):
    head = "| ID | Level | Claim | Evidence | Gate |\n|----|----|----|----|----|\n"
    (project / "docs" / "CLAIMS.md").write_text(head + "".join(f"| {r} |\n" for r in rows), encoding="utf-8")


def test_claim_levels_cannot_be_raised_by_agent(project):
    write_ledger(project, ["C-1 | REFUTED | thing works | audit shows control matches | none"])
    run([".integrity/claims_check.py", "--snapshot"], project, env={"INTEGRITY_HUMAN_APPROVED": "1"})
    assert gates(project, "--only", "claims").returncode == 0
    write_ledger(project, ["C-1 | CONTROLLED | thing works | fake | tests/x.py::t"])        # agent raises level
    r = gates(project, "--only", "claims")
    assert r.returncode == 1 and "raised" in r.stdout
    write_ledger(project, ["C-1 | REFUTED | thing works | audit | none", "C-2 | GENERALIZES | new | e | t"])
    assert "new row at GENERALIZES" in gates(project, "--only", "claims").stdout
    write_ledger(project, ["C-1 | UNAUDITED | thing works | n/a | none"])                  # lowering is allowed
    assert gates(project, "--only", "claims").returncode == 0


def test_ledger_gates_must_exist(project):
    (project / "tests").mkdir(exist_ok=True)
    (project / "tests" / "test_real.py").write_text("def test_exists():\n    pass\n", encoding="utf-8")
    write_ledger(project, ["C-1 | REFUTED | x | audit | test_exists", "C-2 | REFUTED | y | audit | test_real.py"])
    assert gates(project, "--only", "claims").returncode == 0
    write_ledger(project, ["C-1 | REFUTED | x | audit | test_does_not_exist"])
    r = gates(project, "--only", "claims")
    assert r.returncode == 1 and "test_does_not_exist" in r.stdout


def test_status_doc_cannot_overstate_ledger(project):
    write_ledger(project, ["B-1 | REFUTED | x | e | none"])
    run([".integrity/claims_check.py", "--snapshot"], project, env={"INTEGRITY_HUMAN_APPROVED": "1"})
    cfg = json.loads((project / ".integrity" / "config.json").read_text(encoding="utf-8"))
    cfg["status_docs"] = ["docs/BOARD.md"]
    (project / ".integrity" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (project / "docs" / "BOARD.md").write_text("| **B-1** | X solves it | DONE |\n", encoding="utf-8")
    r = gates(project, "--only", "claims")
    assert r.returncode == 1 and "missing [LEVEL] tag" in r.stdout
    (project / "docs" / "BOARD.md").write_text("| **B-1** | X solves it | DONE [CONTROLLED] |\n", encoding="utf-8")
    assert "tagged [CONTROLLED] but ledger says REFUTED" in gates(project, "--only", "claims").stdout
    (project / "docs" / "BOARD.md").write_text("| **B-1** | X solves it | DONE [REFUTED] |\n", encoding="utf-8")
    assert gates(project, "--only", "claims").returncode == 0


def test_project_tests_and_tracked_file_protection(project):
    (project / "evidence.json").write_text('{"x": 1}', encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(project), check=False)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "init"], cwd=str(project), check=False)
    cfg = json.loads((project / ".integrity" / "config.json").read_text(encoding="utf-8"))
    cfg["full_tests"] = ["python -c \"open('evidence.json','w').write('{}')\""]   # a 'test' overwriting tracked evidence
    (project / ".integrity" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    r = gates(project, "--only", "tracked", "--full")
    assert r.returncode == 1 and "evidence.json" in r.stdout and "modified tracked files" in r.stdout
    cfg["tracked_write_allowlist"] = ["evidence.json"]
    (project / ".integrity" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    subprocess.run(["git", "checkout", "--", "evidence.json"], cwd=str(project), check=False)
    assert gates(project, "--only", "tracked", "--full").returncode == 0


def test_reinstall_preserves_project_notes_and_is_idempotent(project):
    p = project / "GEMINI.md"
    p.write_text(p.read_text(encoding="utf-8") + "\n## My project notes\nkeep me\n", encoding="utf-8")
    r = run([str(KIT / "install.py"), str(project), "--relock"], KIT)
    assert r.returncode == 0, r.stdout + r.stderr
    text = p.read_text(encoding="utf-8")
    assert "keep me" in text and text.count("# Research & Engineering Integrity Rules") == 1
    assert gates(project, "--fast").returncode == 0


def test_existing_gemini_md_without_marker_is_preserved(tmp_path):
    (tmp_path / "GEMINI.md").write_text("# Old instructions\nbe nice\n", encoding="utf-8")
    assert run([str(KIT / "install.py"), str(tmp_path), "--no-init"], KIT).returncode == 0
    text = (tmp_path / "GEMINI.md").read_text(encoding="utf-8")
    assert "be nice" in text and "Research & Engineering Integrity Rules" in text
