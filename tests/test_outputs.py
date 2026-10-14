import pytest

from grl_yolo.outputs import clear_previous_results


def results(root):
    file = root / "runs/detect/train/results.csv"
    file.parent.mkdir(parents=True)
    file.write_text("previous run")
    return file


def test_new_experiment_clears_only_default_detection_outputs(tmp_path):
    old = results(tmp_path)
    unrelated = tmp_path / "runs/classify/results.csv"
    unrelated.parent.mkdir()
    unrelated.write_text("keep")
    assert clear_previous_results({}, root=tmp_path)
    assert not old.exists()
    assert unrelated.read_text() == "keep"


@pytest.mark.parametrize("overrides,baseline", [({}, True), ({"resume": True}, False),
                                                ({"project": "custom"}, False)])
def test_baseline_resume_and_custom_outputs_are_preserved(tmp_path, overrides, baseline):
    old = results(tmp_path)
    assert not clear_previous_results(overrides, baseline, tmp_path)
    assert old.read_text() == "previous run"


def test_previous_checkpoint_used_as_initialization_is_not_deleted(tmp_path):
    old = results(tmp_path)
    with pytest.raises(ValueError, match="Training input"):
        clear_previous_results({"pretrained": str(old.parent / "weights/best_detector.pt")}, root=tmp_path)
    assert old.exists()
