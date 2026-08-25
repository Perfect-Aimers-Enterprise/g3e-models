import json
from pathlib import Path

from shared.validation import validate_semantic_file, validate_semantic_directory


def _write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_valid_file_has_no_problems(tmp_path):
    p = tmp_path / "a.json"
    _write(p, {"image": "a.jpg", "objects": [], "semantic": {
        "state": "normal", "severity": "none", "description": "x", "reason": "",
    }})
    data, problems = validate_semantic_file(p)
    assert data is not None
    assert problems == []


def test_missing_required_field_is_caught(tmp_path):
    p = tmp_path / "a.json"
    _write(p, {"image": "a.jpg", "objects": [], "semantic": {
        "state": "normal", "severity": "none", "description": "x",
    }})
    data, problems = validate_semantic_file(p)
    assert any("reason" in pr for pr in problems)


def test_invalid_state_value_is_caught(tmp_path):
    p = tmp_path / "a.json"
    _write(p, {"image": "a.jpg", "objects": [], "semantic": {
        "state": "definitely_not_a_real_state", "severity": "none", "description": "x", "reason": "",
    }})
    data, problems = validate_semantic_file(p)
    assert any("invalid state" in pr for pr in problems)


def test_invalid_severity_value_is_caught(tmp_path):
    p = tmp_path / "a.json"
    _write(p, {"image": "a.jpg", "objects": [], "semantic": {
        "state": "normal", "severity": "extremely_bad", "description": "x", "reason": "",
    }})
    data, problems = validate_semantic_file(p)
    assert any("invalid severity" in pr for pr in problems)


def test_malformed_json_returns_none_and_a_problem(tmp_path):
    p = tmp_path / "a.json"
    p.write_text("{not valid json")
    data, problems = validate_semantic_file(p)
    assert data is None
    assert len(problems) == 1


def test_missing_semantic_key_entirely(tmp_path):
    p = tmp_path / "a.json"
    _write(p, {"image": "a.jpg", "objects": []})
    data, problems = validate_semantic_file(p)
    assert data is None


def test_empty_description_is_caught_but_empty_reason_is_allowed(tmp_path):
    p = tmp_path / "a.json"
    _write(p, {"image": "a.jpg", "objects": [], "semantic": {
        "state": "normal", "severity": "none", "description": "", "reason": "",
    }})
    data, problems = validate_semantic_file(p)
    assert any("empty field 'description'" in pr for pr in problems)
    assert not any("reason" in pr for pr in problems)


def test_validate_directory_aggregates_state_counts_and_duplicates(tmp_path):
    d = tmp_path / "semantic"
    for i in range(3):
        _write(d / f"n{i}.json", {"image": f"n{i}.jpg", "objects": [], "semantic": {
            "state": "normal", "severity": "none", "description": "same desc", "reason": "",
        }})
    _write(d / "t0.json", {"image": "t0.jpg", "objects": [], "semantic": {
        "state": "potential_threat", "severity": "high", "description": "unique desc", "reason": "r",
    }})

    report = validate_semantic_directory(d)
    assert report.total_checked == 4
    assert report.ok is True
    assert report.state_counts == {"normal": 3, "potential_threat": 1}
    assert report.duplicate_description_count == 3  # all 3 "same desc" copies


def test_validate_directory_reports_issues_from_bad_files(tmp_path):
    d = tmp_path / "semantic"
    _write(d / "bad.json", {"image": "bad.jpg", "objects": [], "semantic": {
        "state": "not_real", "severity": "none", "description": "x", "reason": "",
    }})
    report = validate_semantic_directory(d)
    assert report.ok is False
    assert len(report.issues) == 1


def test_render_includes_duplicate_warning_when_present(tmp_path):
    d = tmp_path / "semantic"
    for i in range(2):
        _write(d / f"n{i}.json", {"image": f"n{i}.jpg", "objects": [], "semantic": {
            "state": "normal", "severity": "none", "description": "same", "reason": "",
        }})
    report = validate_semantic_directory(d)
    rendered = report.render()
    assert "repeated across multiple images" in rendered
