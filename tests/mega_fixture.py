"""Portable synthetic speech and explicit offline model/environment fixtures."""

import os
from pathlib import Path
import subprocess
import wave


def settings(root, **overrides):
    assets = os.environ.get("MEGA_ASR_TEST_ASSETS", "")
    python = os.environ.get("MEGA_ASR_TEST_PYTHON", "")
    assert assets and Path(assets).is_dir(), "Prepare the pinned model fixture and set MEGA_ASR_TEST_ASSETS"
    assert python and Path(python).is_file(), "Set MEGA_ASR_TEST_PYTHON to the dedicated inference environment"
    return {"asset_directory": assets, "environment_python": python, "allowed_root": str(root),
        "language": "English", "timeout_sec": 180, **overrides}


def speech(root, *, opus=False, repeat=1):
    wav = root / "speech.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "flite=text=The weather is nice today.:voice=slt",
        "-ac", "1", "-ar", "16000", str(wav)], check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if repeat > 1:
        with wave.open(str(wav), "rb") as source:
            data = source.readframes(source.getnframes())
        with wave.open(str(wav), "wb") as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000); output.writeframes(data * repeat)
    if not opus:
        return wav
    target = root / "speech.ogg"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-c:a", "libopus", str(target)],
        check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return target


def alive(pid):
    try:
        return Path("/proc", str(pid), "stat").read_text().split()[2] not in {"Z", "X"}
    except (FileNotFoundError, ProcessLookupError):
        return False


def worker_pids(job_dirs):
    """Observe this test's actual Linux model jobs without an extra test dependency."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            if command and command[-1].endswith(b"/worker.py") and os.readlink(entry / "cwd") in job_dirs:
                found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return found
