"""FB1/FB2/FB3/FB4: real MEGA CPU inference, validation, immutable assets and process fencing."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import wave
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(Path(__file__).parent)]
from blocs.mega_asr import logic, assets_store, worker
from mega_fixture import settings, speech, alive


def refused(action, contains=None):
    try:
        action()
    except (logic.MegaError, assets_store.AssetError) as exc:
        assert contains is None or contains in str(exc), exc
        return
    raise AssertionError("Invalid operation was accepted")


def parent_death(root):
    """An abruptly killed host must not leave the guardian's native child running."""
    pidfile = root / "native-pid"
    source = "\n".join([
        "import sys,time,types;from pathlib import Path",
        "p=types.ModuleType('mega_parent');p.__path__=[sys.argv[1]];sys.modules[p.__name__]=p",
        "from mega_parent.logic import run_process",
        "code=\"import os,time;from pathlib import Path;Path(__import__('sys').argv[1]).write_text(str(os.getpid()));time.sleep(30)\"",
        "run_process([sys.executable,'-B','-c',code,sys.argv[2]],directory=Path(sys.argv[2]).parent,deadline=time.monotonic()+30,cancel=lambda:False,limit=1024)",
    ])
    parent = subprocess.Popen([sys.executable, "-E", "-B", "-c", source, str(Path(logic.__file__).parent), str(pidfile)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child = None
    try:
        deadline = time.monotonic() + 5
        while not pidfile.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert pidfile.exists(), "Owned native fixture did not start"
        child = int(pidfile.read_text())
        parent.kill(); parent.wait(timeout=3)
        deadline = time.monotonic() + 3
        while alive(child) and time.monotonic() < deadline:
            time.sleep(.02)
        assert not alive(child), "Native process survived abrupt host death"
    finally:
        if parent.poll() is None:
            parent.kill(); parent.wait(timeout=3)
        if child and alive(child):
            os.kill(child, signal.SIGKILL)


def duration_limits(root):
    # The duration limit is expressed at 16 kHz AFTER resampling. In particular,
    # Opus decodes at 48 kHz: trimming first silently shortens a long recording.
    for rate in (8000, 16000, 44100, 48000):
        path = root / f"rate-{rate}.wav"
        for seconds in (1, 2):
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1); output.setsampwidth(2); output.setframerate(rate)
                output.writeframes(b"\0\0" * (rate * seconds))
            decode = lambda: logic.decode_audio(path, root, 1, time.monotonic()+5, lambda:False)
            if seconds == 1:
                assert len(decode()) == 32000, rate
            else:
                refused(decode, "duration limit")
    path = root / "one-extra-sample.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
        output.writeframes(b"\0\0" * 16001)
    refused(lambda: logic.decode_audio(path,root,1,time.monotonic()+5,lambda:False), "duration limit")


def main():
    for value in ({"api_key": "no"}, {"language": []}, {"context": "\ud800"}, {"device": "auto"},
        {"threads": True}, {"precision": "int4"}, {"quality_threshold": float("nan")}, {"max_duration_sec": 1801}):
        refused(lambda: logic.configuration(value))
    for value in (None, "", {"action": "stop"}, '{"event":"x","event":"recording_ready"}'):
        refused(lambda: logic.request(value))
    from urllib.request import Request
    for url in ("http://huggingface.co/model", "https://other.invalid/file", "https://huggingface.co.evil.invalid/file"):
        refused(lambda: assets_store.PinnedRedirects().redirect_request(Request("https://huggingface.co/model"), None, 302, "", {}, url))
    # Completion gating rejects a truncated decode even when the SDK returned text.
    sequence = type("Sequence", (), {"shape": (1,33), "__getitem__": lambda self,key: 5})()
    fake = SimpleNamespace(generate=lambda **kw: SimpleNamespace(sequences=sequence))
    worker.guard_generation(fake, [9], 32)
    refused(lambda: fake.generate(input_ids=SimpleNamespace(shape=(1,1))), "token_limit")
    with TemporaryDirectory(prefix="mega-real-") as directory:
        root = Path(directory)
        config = settings(root)
        storage = root / "storage"; storage.mkdir()
        payload = root / "tiny"; payload.write_bytes(b"test")
        spec = {"path": "tiny", "bytes": 4, "sha256": hashlib.sha256(b"test").hexdigest()}
        assert assets_store.verify(root, spec)
        payload.write_bytes(b"BAD!"); refused(lambda: assets_store.verify(root, spec))
        link = root / "linked"; link.symlink_to(payload)
        refused(lambda: assets_store.verify(root, {**spec, "path": "linked"}))
        before = time.monotonic()
        refused(lambda: logic.run_process([sys.executable,"-c","import time;time.sleep(30)"],directory=root,
            deadline=before+3,cancel=lambda:time.monotonic()-before>.15,limit=1024))
        assert time.monotonic()-before < 1.5
        refused(lambda: logic.run_process([sys.executable,"-c","print('x'*10000)"],directory=root,
            deadline=time.monotonic()+3,cancel=lambda:False,limit=1024))
        parent_death(root)
        duration_limits(root)
        audio = speech(root, opus=True, repeat=3)
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        refused(lambda: logic.transcribe(str(audio), {**config,"max_duration_sec":1}, root, storage))
        refused(lambda: logic.transcribe(str(audio), {**config,"allowed_root":str(storage)}, root, storage))
        refused(lambda: logic.transcribe(str(audio), {**config,"asset_directory":""}, root, storage))
        event = {"event":"recording_ready","recording_id":"clip-1","stream_id":"source-1","call_id":"call-1",
            "path":str(audio),"bytes":audio.stat().st_size}
        refused(lambda: logic.transcribe({**event,"bytes":1},config,root,storage))
        result = logic.transcribe(event, {**config,"routing":"mega","chunk_duration_sec":5}, root, storage)
        assert result["final"] and len(result["chunks"]) == 2, result
        assert all(c["use_lora"] and c["route_source"]=="default" for c in result["chunks"]), result
        assert "weather" in result["text"].lower() and "nice" in result["text"].lower(), result
        assert not result["timestamps_available"] and result["source"]["call_id"]=="call-1"
        assert result["source"]["sha256"]==digest and hashlib.sha256(audio.read_bytes()).hexdigest()==digest
        assert not list(storage.iterdir()), "Normal execution leaked private snapshots"
        print("[ok] Actual MEGA LoRA CPU/Ogg multi-chunk transcription:",result["text"],flush=True)
        # Wrong metadata or incomplete chunk coverage cannot be promoted to final output.
        raw = {k:result[k] for k in ("text","chunks")}
        raw["chunks"][0]["end_sec"] = .1
        refused(lambda: logic.validate_result(json.dumps(raw),result["duration_sec"],logic.configuration({**config,"routing":"mega","chunk_duration_sec":5})))
    print("[ok] MEGA validation, pinned cache integrity, true LoRA inference and cancellation/parent-death fencing")


if __name__ == "__main__":
    main()
