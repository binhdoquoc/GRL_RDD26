"""Stream subprocess errors into notebook output and preserve a complete log."""
import subprocess
from collections import deque
from pathlib import Path


def run_logged(arguments, cwd, env, log_file):
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=30)
    process = subprocess.Popen(
        list(map(str, arguments)), cwd=cwd, env={**env, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
    )
    try:
        with log_file.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
        code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
    if code:
        raise RuntimeError(f"Subprocess exited with code {code}. Full output: {log_file}\n"
                           f"Last output lines:\n{''.join(tail)}")
    return log_file
