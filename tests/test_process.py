import os
import sys

import pytest

from grl_yolo.process import run_logged


def test_child_traceback_is_displayed_and_saved(tmp_path, capsys):
    script = "print('training started'); raise FileNotFoundError('source image example.jpg')"
    log = tmp_path / "logs/train.log"
    with pytest.raises(RuntimeError) as error:
        run_logged([sys.executable, "-c", script], tmp_path, dict(os.environ), log)
    assert "FileNotFoundError: source image example.jpg" in str(error.value)
    assert "FileNotFoundError: source image example.jpg" in log.read_text()
    assert "training started" in capsys.readouterr().out


def test_success_streams_both_stdout_and_stderr(tmp_path, capsys):
    script = "import sys; print('stdout message'); print('stderr message', file=sys.stderr)"
    log = tmp_path / "success.log"
    run_logged([sys.executable, "-c", script], tmp_path, dict(os.environ), log)
    printed = capsys.readouterr().out
    assert "stdout message" in printed and "stderr message" in printed
    assert log.read_text() == printed
