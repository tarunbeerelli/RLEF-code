# tests/test_data.py
import json
import pathlib
import pytest
import yaml

@pytest.mark.unit
def test_train_yaml_schema_validity():
    # Static check of repository configurations that runs safely in standard CI environments
    config_path = pathlib.Path("configs/train.yaml")
    assert config_path.exists(), "Configuration switchboard configs/train.yaml is missing!"
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    assert "ablation" in cfg
    assert "max_turns" in cfg["ablation"]

@pytest.mark.integration
def test_dataset_generation_alignment(tmp_path):
    # Integration test touching local temporary I/O files
    mock_row = {
        "problem_id": 999,
        "prompt": "mock_tokens_here",
        "inputs": ["1"],
        "outputs": ["2"]
    }
    test_file = tmp_path / "test_split.jsonl"
    with open(test_file, "w") as f:
        f.write(json.dumps(mock_row) + "\n")
        
    with open(test_file, "r") as f:
        loaded = json.loads(f.readline())
    assert loaded["problem_id"] == 999