"""Regression checks for setup on kernels missing ensurepip or using Python 3.13."""
import os
import subprocess
import venv
from pathlib import Path

import pytest

from grl_yolo import environment


def test_reuses_real_python312_environment_without_pip(tmp_path, monkeypatch):
    directory = tmp_path / "env"
    # A real environment deliberately created without pip/ensurepip bootstrapping.
    venv.EnvBuilder(with_pip=False).create(directory)
    monkeypatch.setattr(environment, "find_uv", lambda directory: Path("fake-uv"))
    monkeypatch.setenv("PYTHONPATH", "/kernel/python3.13/site-packages")
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    python, env = environment.prepare_environment(tmp_path, directory, install_packages=False)
    assert python == environment.environment_python(directory)
    assert environment.python_version(python, env) == (3, 12)
    assert "PYTHONPATH" not in env
    assert env["MPLBACKEND"] == "Agg"
    probe = subprocess.run([str(python), "-m", "pip", "--version"], env=env, capture_output=True)
    assert probe.returncode != 0  # setup does not require pip in the environment


def test_recovers_partial_environment_without_deleting_it(tmp_path, monkeypatch):
    directory = tmp_path / "partial-env"
    directory.mkdir()
    sentinel = directory / "keep.txt"
    sentinel.write_text("existing files must survive")
    commands = []
    original_run = subprocess.run
    monkeypatch.setattr(environment, "find_uv", lambda directory: Path("fake-uv"))

    def run(arguments, **kwargs):
        if arguments[0] == "fake-uv":
            commands.append(arguments)
            assert arguments[1:4] == ["venv", "--python", "3.12"]
            # Model uv's result with a real, unseeded environment; no network required.
            venv.EnvBuilder(with_pip=False).create(arguments[4])
            return subprocess.CompletedProcess(arguments, 0)
        return original_run(arguments, **kwargs)

    monkeypatch.setattr(environment.subprocess, "run", run)
    python, env = environment.prepare_environment(tmp_path, directory, install_packages=False)
    assert python != environment.environment_python(directory)
    assert environment.python_version(python, env) == (3, 12)
    assert sentinel.read_text() == "existing files must survive"
    assert len(commands) == 1


def test_incompatible_python313_is_not_reused(tmp_path, monkeypatch):
    directory = tmp_path / "incompatible-env"
    directory.mkdir()
    (directory / "pyvenv.cfg").write_text("version = 3.13")
    commands = []
    monkeypatch.setattr(environment, "find_uv", lambda directory: Path("fake-uv"))
    original = environment.environment_python(directory)
    monkeypatch.setattr(environment, "python_version", lambda python, env: (3, 13) if python == original else (3, 12))

    def run(arguments, **kwargs):
        commands.append(arguments)
        created = Path(arguments[-1])
        created.mkdir()
        (created / "pyvenv.cfg").write_text("version = 3.12")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(environment.subprocess, "run", run)
    python, _ = environment.prepare_environment(tmp_path, directory, install_packages=False)
    assert python != original
    assert commands[0][1:4] == ["venv", "--python", "3.12"]
    assert all("ensurepip" not in " ".join(command) for command in commands)


def test_packages_install_via_uv_with_pypi_index(tmp_path, monkeypatch):
    directory = tmp_path / "env"
    directory.mkdir()
    (directory / "pyvenv.cfg").write_text("version = 3.12")
    monkeypatch.setattr(environment, "find_uv", lambda directory: Path("fake-uv"))
    monkeypatch.setattr(environment, "python_version", lambda python, env: (3, 12))
    commands = []

    def run(arguments, **kwargs):
        commands.append(arguments)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(environment.subprocess, "run", run)
    python, _ = environment.prepare_environment(tmp_path, directory)
    assert len(commands) == 2
    assert all(command[:4] == ["fake-uv", "pip", "install", "--python"] for command in commands)
    assert commands[0][-1] == "https://pypi.org/simple"
    assert commands[1][-4:] == ["-r", str(tmp_path / "requirements.txt"), "--index-url", environment.PYPI_INDEX]


def test_pytorch_index_timeout_falls_back_to_pypi(tmp_path, monkeypatch):
    commands = []
    custom = "https://download.pytorch.org/whl/cu124"

    def run(arguments, **kwargs):
        commands.append(arguments)
        if arguments[-1] == custom:
            raise subprocess.CalledProcessError(2, arguments)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(environment.subprocess, "run", run)
    environment.install_training_packages("uv", "/tmp/env/bin/python", tmp_path, {}, torch_index=custom)
    assert len(commands) == 3
    assert commands[0][-1] == custom
    assert commands[1][-1] == environment.PYPI_INDEX
    assert "torch==2.5.1" in commands[1] and "torchvision==0.20.1" in commands[1]
    assert commands[2][-1] == environment.PYPI_INDEX


def test_pypi_failure_stops_with_clear_error(tmp_path, monkeypatch):
    def run(arguments, **kwargs):
        raise subprocess.CalledProcessError(2, arguments)

    monkeypatch.setattr(environment.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Check Kaggle Internet access"):
        environment.install_training_packages("uv", "/tmp/env/bin/python", tmp_path, {})


def test_invalid_interpreter_is_detected(tmp_path):
    broken = tmp_path / "python"
    broken.write_bytes(b"not an executable")
    assert environment.python_version(broken, dict(os.environ)) is None


def test_inherited_inline_backend_does_not_break_imports(tmp_path, monkeypatch):
    # Use the real test interpreter with installed plotting/training dependencies.
    import sys
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    directory = tmp_path / "env"
    directory.mkdir()
    (directory / "pyvenv.cfg").write_text("version = 3.12")
    monkeypatch.setattr(environment, "find_uv", lambda directory: Path("fake-uv"))
    monkeypatch.setattr(environment, "python_version", lambda python, env: (3, 12))
    _, env = environment.prepare_environment(tmp_path, directory, install_packages=False)
    result = subprocess.run([sys.executable, "-c",
                             "import matplotlib, ultralytics; "
                             "assert matplotlib.get_backend().lower() == 'agg'; print('imports OK')"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "imports OK" in result.stdout
