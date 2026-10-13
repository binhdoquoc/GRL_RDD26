"""Bootstrap Kaggle training without kernel pip, ensurepip or Python-version coupling."""
import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import uuid
from pathlib import Path


PYPI_INDEX = "https://pypi.org/simple"


def install_training_packages(uv, python, project, env, torch_index=PYPI_INDEX):
    """Install pinned packages via PyPI; fall back there if a custom index fails."""
    indexes = list(dict.fromkeys([torch_index or PYPI_INDEX, PYPI_INDEX]))
    base = [str(uv), "pip", "install", "--python", str(python)]
    for index, url in enumerate(indexes):
        print("Installing pinned PyTorch from:", url, flush=True)
        try:
            subprocess.run([*base, "torch==2.5.1", "torchvision==0.20.1", "--index-url", url],
                           env=env, check=True)
            break
        except subprocess.CalledProcessError as error:
            if index + 1 == len(indexes):
                raise RuntimeError(
                    "PyTorch installation failed from PyPI. Check Kaggle Internet access and the "
                    "installer output above. Training cannot start until the pinned packages are installed."
                ) from error
            print(f"Index {url} failed; retrying from PyPI.", flush=True)
    subprocess.run([*base, "-r", str(Path(project) / "requirements.txt"), "--index-url", PYPI_INDEX],
                   env=env, check=True)


def find_uv(directory):
    """Use an installed uv or download its official Linux standalone executable."""
    installed = shutil.which("uv")
    if installed:
        return Path(installed)
    directory = Path(directory)
    binary = directory / "uv"
    if binary.is_file():
        return binary
    if platform.system() != "Linux":
        raise RuntimeError("This bootstrap targets Kaggle Linux. Install uv locally to use it elsewhere.")
    architecture = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64"}.get(platform.machine().lower())
    if not architecture:
        raise RuntimeError(f"Unsupported Linux architecture: {platform.machine()}")
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "uv.tar.gz"
    url = f"https://github.com/astral-sh/uv/releases/latest/download/uv-{architecture}-unknown-linux-gnu.tar.gz"
    print("Downloading standalone uv:", url, flush=True)
    # Copy only the named regular executable; do not extract arbitrary archive paths.
    request = urllib.request.Request(url, headers={"User-Agent": "yolo11s-grl-notebook"})
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    with tarfile.open(archive, "r:gz") as bundle:
        members = [member for member in bundle.getmembers() if member.isfile() and Path(member.name).name == "uv"]
        if len(members) != 1:
            raise RuntimeError("Official uv archive did not contain one regular uv executable")
        pending = directory / "uv.pending"
        with bundle.extractfile(members[0]) as source, pending.open("wb") as output:
            shutil.copyfileobj(source, output)
    pending.chmod(0o755)
    pending.replace(binary)
    return binary


def environment_python(directory):
    return Path(directory) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def python_version(python, env):
    if not Path(python).is_file():
        return None
    try:
        result = subprocess.run(
            [str(python), "-c", "import json, sys; print(json.dumps(list(sys.version_info[:2])))"],
            env=env, text=True, capture_output=True,
        )
    except OSError:
        return None
    if result.returncode:
        return None
    try:
        return tuple(json.loads(result.stdout.strip()))
    except (ValueError, TypeError):
        return None


def prepare_environment(project, directory="/tmp/yolo11s-grl-py312", install_packages=True,
                        torch_index=PYPI_INDEX):
    """Return Python 3.12 and its environment, recovering from incomplete setup."""
    project, directory = Path(project).resolve(), Path(directory).resolve()
    env = dict(os.environ)
    # Kernel paths may reference Python 3.13 libraries, which must not leak here.
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    # Kaggle's kernel may export an inline backend unavailable in isolated Python.
    # All training/validation plots are saved to files and displayed by the kernel.
    env["MPLBACKEND"] = "Agg"
    env["UV_CACHE_DIR"] = str(directory.parent / "yolo11s-grl-uv-cache")
    env["UV_PYTHON_INSTALL_DIR"] = str(directory.parent / "yolo11s-grl-python")
    uv = find_uv(directory.parent / "yolo11s-grl-tools")
    python = environment_python(directory)
    if not (directory / "pyvenv.cfg").is_file() or python_version(python, env) != (3, 12):
        if directory.exists():
            # Leave existing/partial environments intact and create a fresh task-owned one.
            directory = directory.with_name(f"{directory.name}-{uuid.uuid4().hex[:8]}")
            python = environment_python(directory)
        print("Creating isolated Python 3.12 environment:", directory, flush=True)
        subprocess.run([str(uv), "venv", "--python", "3.12", str(directory)], env=env, check=True)
        if not (directory / "pyvenv.cfg").is_file() or python_version(python, env) != (3, 12):
            raise RuntimeError("Environment setup did not produce a working Python 3.12 interpreter")
    if install_packages:
        # Linux PyPI wheels include CUDA dependencies; no PyTorch-hosted index is required.
        install_training_packages(uv, python, project, env, torch_index)
    env["YOLO_CONFIG_DIR"] = str(project / ".ultralytics")
    env["MPLCONFIGDIR"] = str(project / ".matplotlib")
    print("Training Python:", python, flush=True)
    return python, env
