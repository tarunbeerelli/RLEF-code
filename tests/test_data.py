import pathlib

import pytest
from rlef.data import APPSProblem, difficulty_split, load_apps_split

DATA_DIR = pathlib.Path("data/raw/APPS")


def skip_if_no_data():
    return pytest.mark.skipif(
        not DATA_DIR.exists(),
        reason="APPS data not downloaded — run scripts/download_apps.sh",
    )


@skip_if_no_data()
def test_load_train_returns_problems():
    problems = load_apps_split(DATA_DIR, split="train")
    assert len(problems) > 700


@skip_if_no_data()
def test_problem_fields_populated():
    problems = load_apps_split(DATA_DIR, split="train")
    p = problems[0]
    assert isinstance(p, APPSProblem)
    assert len(p.question) > 10
    assert len(p.inputs) > 0
    assert len(p.outputs) > 0
    assert len(p.inputs) == len(p.outputs)
    assert p.difficulty in ["introductory", "interview", "competition"]


@skip_if_no_data()
def test_difficulty_filter():
    problems = load_apps_split(DATA_DIR, split="train", difficulties=["introductory"])
    assert all(p.difficulty == "introductory" for p in problems)
    assert len(problems) > 0


@skip_if_no_data()
def test_difficulty_split_buckets():
    problems = load_apps_split(DATA_DIR, split="train")
    buckets = difficulty_split(problems)
    assert set(buckets.keys()) == {"introductory", "interview", "competition"}
    total = sum(len(v) for v in buckets.values())
    assert total == len(problems)


@skip_if_no_data()
def test_no_problems_missing_test_cases():
    problems = load_apps_split(DATA_DIR, split="train")
    for p in problems:
        assert len(p.inputs) > 0
        assert len(p.outputs) > 0


def test_load_raises_on_bad_path():
    with pytest.raises(FileNotFoundError):
        load_apps_split("/nonexistent/path", split="train")
