"""Completed-file local MEGA-ASR adapter: no source URLs, implicit downloads or partial output."""

from collections.abc import Mapping
from contextlib import suppress
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
import wave

DEFAULTS = {
    "asset_directory": "", "environment_python": "", "allowed_root": "",
    "device": "cpu", "precision": "bfloat16", "language": "auto", "context": "",
    "routing": "auto", "quality_threshold": 0.5, "threads": 2,
    "chunk_duration_sec": 20, "max_duration_sec": 300, "max_file_bytes": 67108864,
    "max_new_tokens": 512, "timeout_sec": 600, "memory_limit_mib": 16384,
}
FORMATS = {".wav", ".ogg", ".opus", ".webm", ".mp3", ".m4a", ".flac", ".mp4"}


class MegaError(ValueError):
    """Owned diagnostic only, never an arbitrary native exception or input content."""


class Cancelled(MegaError):
    pass


def configuration(raw):
    if not isinstance(raw, Mapping) or set(raw) - set(DEFAULTS) - {"execution", "position", "runtime_path", "runtime_path_label"}:
        raise MegaError("Unsupported MEGA ASR setting.")
    cfg = {**DEFAULTS, **{k: v for k, v in raw.items() if k in DEFAULTS}}
    for key in ("asset_directory", "environment_python", "allowed_root"):
        value = cfg[key]
        if not isinstance(value, str) or len(value) > 4096 or any(ord(c) < 32 for c in value):
            raise MegaError("Choose valid local paths in the settings.")
    for key, choices in {
        "device": ("cpu", "cuda:0"), "precision": ("float32", "bfloat16"),
        "language": ("auto", "English", "Chinese"), "routing": ("auto", "mega"),
    }.items():
        if not isinstance(cfg[key], str) or cfg[key] not in choices:
            raise MegaError("Unsupported device, precision, language or routing mode.")
    for key, low, high in (
        ("threads", 1, 16), ("chunk_duration_sec", 5, 30), ("max_duration_sec", 1, 1800),
        ("max_file_bytes", 1024, 268435456), ("max_new_tokens", 32, 2048),
        ("timeout_sec", 1, 3600), ("memory_limit_mib", 2048, 65536),
    ):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise MegaError(f"{key} must be an integer between {low} and {high}.")
    value = cfg["quality_threshold"]
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise MegaError("The acoustic routing threshold must be between 0 and 1.")
    value = cfg["context"]
    try:
        valid = isinstance(value, str) and len(value.encode("utf-8")) <= 4096 and "\x00" not in value
        for key in ("asset_directory", "environment_python", "allowed_root"):
            cfg[key].encode("utf-8")
    except UnicodeError:
        valid = False
    if not valid:
        raise MegaError("Use valid UTF-8 paths and a context of at most 4 KiB.")
    return cfg


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise MegaError("Duplicate JSON property.")
            result[key] = value
        return result
    try:
        if len(raw.encode("utf-8")) > 16384:
            raise MegaError("Input exceeds 16 KiB.")
        return json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(MegaError("Finite JSON required.")))
    except (ValueError, UnicodeError, RecursionError):
        raise MegaError("Expected a file path or a valid recording_ready event.") from None


def request(raw):
    """A finished-file event is distinct from a microphone start/stop command."""
    if isinstance(raw, str) and raw.lstrip().startswith("{"):
        raw = strict_json(raw)
    metadata = {}
    if isinstance(raw, dict):
        if raw.get("event") != "recording_ready":
            raise MegaError("Send recording_ready after file finalization, not a stream start/stop command.")
        for key in ("recording_id", "stream_id", "call_id"):
            if key == "call_id" and key not in raw:
                continue
            value = raw.get(key)
            if not isinstance(value, str) or not 1 <= len(value) <= 128 or any(ord(char) < 32 for char in value):
                raise MegaError("Invalid recording correlation.")
            metadata[key] = value
        if type(raw.get("bytes")) is not int or raw["bytes"] < 1:
            raise MegaError("recording_ready requires the finalized file size.")
        metadata["bytes"] = raw["bytes"]
        raw = raw.get("path")
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise MegaError("recording_ready requires an absolute file path.")
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 4096 or any(ord(char) < 32 for char in raw):
        raise MegaError("A nonempty local audio file path is required.")
    return raw.strip(), metadata



def check(deadline, cancel):
    if cancel():
        raise Cancelled("Transcription cancelled; no partial text was published.")
    if time.monotonic() >= deadline:
        raise MegaError("Operation deadline exceeded; no partial text was published.")


ERRORS = {
    "assets_missing": "MEGA ASR assets are missing; run the explicit setup helper and configure their directory.",
    "assets_corrupt": "A pinned MEGA ASR file failed verification; it was not loaded or overwritten.",
    "dependency": "The selected Python environment lacks the pinned inference dependencies.",
    "device_unavailable": "The configured CPU/CUDA device or numeric precision is unavailable.",
    "model_failed": "MEGA ASR could not load or transcribe the recording; check resources and prerequisites.",
    "token_limit": "A chunk reached the token limit; shorten chunks or increase the limit. No partial transcript was published.",
    "resource_limit": "MEGA ASR exceeded the configured process memory allowance.",
}


def run_process(args, *, directory, deadline, cancel, limit, stdin=None):
    """Bound every pipe; fence children on cancellation and abrupt framework-host death."""
    check(deadline, cancel)
    if sys.platform != "linux":
        raise MegaError("This package currently requires Linux.")
    envelope = stdin or b""
    if len(envelope) > 32768:
        raise MegaError("Internal settings envelope exceeds its limit.")
    environment = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL"}}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
        HF_HOME=str(directory / "hf-cache"), XDG_CACHE_HOME=str(directory / "cache"),
        NUMBA_CACHE_DIR=str(directory / "numba-cache"), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2")
    command = [sys.executable, "-E", "-B", str(Path(__file__).with_name("process_guard.py")), str(os.getpid()), *args]
    process = subprocess.Popen(command, cwd=directory, env=environment, start_new_session=True,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    selector, output, errors, offset = selectors.DefaultSelector(), bytearray(), bytearray(), 0
    try:
        if process.stdin is not None:
            os.set_blocking(process.stdin.fileno(), False)
            if envelope:
                selector.register(process.stdin, selectors.EVENT_WRITE, "in")
            else:
                process.stdin.close()
        for stream, name in ((process.stdout, "out"), (process.stderr, "err")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            check(deadline, cancel)
            for key, _ in selector.select(timeout=0.03):
                if key.data == "in":
                    try:
                        offset += os.write(key.fd, envelope[offset:offset + 4096])
                    except BrokenPipeError:
                        offset = len(envelope)
                    if offset == len(envelope):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target, maximum = (output, limit) if key.data == "out" else (errors, 262144)
                if len(target) + len(chunk) > maximum:
                    raise MegaError("The audio tool or model exceeded its bounded output size.")
                target.extend(chunk)
        while process.poll() is None:
            check(deadline, cancel)
            time.sleep(0.02)
        check(deadline, cancel)
        if process.returncode:
            try:
                detail = json.loads(output).get("error_code")
            except (ValueError, AttributeError):
                detail = None
            raise MegaError(ERRORS.get(detail, "Audio conversion or local model execution failed."))
        return bytes(output)
    finally:
        selector.close()
        # The guardian owns this group. Fence descendants even after normal exit.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def snapshot(path, allowed, target, maximum, expected, deadline, cancel):
    """Keep the source unchanged and decode an immutable, private local copy."""
    source = (allowed / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not source.is_relative_to(allowed) or source.suffix.lower() not in FORMATS:
        raise MegaError("Audio must be a supported local file inside the permitted directory.")
    descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise MegaError("Audio file is empty, oversized or not a regular file.")
        if expected is not None and before.st_size != expected:
            raise MegaError("The completed recording size differs from its event.")
        digest, size = hashlib.sha256(), 0
        with open(target, "xb") as output:
            os.chmod(target, 0o600)
            while True:
                check(deadline, cancel)
                chunk = stream.read(min(65536, maximum + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum:
                    raise MegaError("Audio file grew beyond its size limit.")
                digest.update(chunk); output.write(chunk)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns) or size != before.st_size:
            raise MegaError("The source changed while it was read; wait for file finalization.")
    return digest.hexdigest(), size



def validate_result(raw, duration, config):
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != {"text", "chunks"}:
            raise ValueError()
        if not isinstance(data["text"], str) or len(data["text"].encode("utf-8")) > 131072:
            raise ValueError()
        chunks = data["chunks"]
        if not isinstance(chunks, list) or not 1 <= len(chunks) <= math.ceil(duration / config["chunk_duration_sec"]):
            raise ValueError()
        previous, texts = 0, []
        for chunk in chunks:
            if set(chunk) != {"start_sec", "end_sec", "text", "language", "use_lora", "degraded_probability", "route_source"}:
                raise ValueError()
            start, end = chunk["start_sec"], chunk["end_sec"]
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)):
                raise ValueError()
            if abs(start - previous) > 0.0001 or not start < end <= duration + 0.0001 or end - start > config["chunk_duration_sec"] + 0.0001:
                raise ValueError()
            previous = end
            if not isinstance(chunk["text"], str) or len(chunk["text"]) > 32768 or not isinstance(chunk["language"], str) or len(chunk["language"]) > 128:
                raise ValueError()
            if type(chunk["use_lora"]) is not bool:
                raise ValueError()
            probability = chunk["degraded_probability"]
            if config["routing"] == "auto":
                if chunk["route_source"] != "router" or type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError()
                if chunk["use_lora"] != (probability >= config["quality_threshold"]):
                    raise ValueError()
            elif chunk["route_source"] != "default" or probability is not None or not chunk["use_lora"]:
                raise ValueError()
            texts.append(chunk["text"].strip())
        if abs(previous - duration) > 0.0001 or data["text"] != "\n".join(t for t in texts if t):
            raise ValueError()
        return data
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise MegaError("MEGA ASR returned an invalid or incomplete result; no partial text was published.") from None


def decode_audio(source, directory, maximum_seconds, deadline, cancel):
    """Count the sentinel sample at the model rate, not the container's source rate."""
    maximum_samples = maximum_seconds * 16000
    audio = run_process(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-max_alloc", "67108864",
        "-format_whitelist", "wav,ogg,matroska,webm,mp3,mov,flac", "-protocol_whitelist", "file,pipe", "-i", str(source),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-af",
        f"aresample=16000,atrim=end_sample={maximum_samples + 1}",
        "-f", "s16le", "pipe:1"], directory=directory, deadline=deadline, cancel=cancel, limit=(maximum_samples + 1) * 2)
    if not audio or len(audio) % 2:
        raise MegaError("The file contains no decodable audio.")
    if len(audio) > maximum_samples * 2:
        raise MegaError("Audio exceeds the configured duration limit; no truncation was transcribed.")
    return audio


def transcribe(raw, config, root, storage, cancel=lambda: False):
    config = configuration(config)
    path, correlation = request(raw)
    root, storage = Path(root), Path(storage)
    if not storage.is_absolute() or storage.is_symlink() or not storage.is_dir():
        raise MegaError("A valid block-owned storage directory is required.")
    def resolve(value):
        candidate = Path(value).expanduser()
        return (root / candidate).absolute() if not candidate.is_absolute() else candidate
    allowed = resolve(config["allowed_root"]).resolve() if config["allowed_root"] else root.resolve()
    if not allowed.is_dir():
        raise MegaError("The permitted audio directory does not exist.")
    if not config["asset_directory"]:
        raise MegaError(ERRORS["assets_missing"])
    assets = resolve(config["asset_directory"])
    if assets.is_symlink() or not assets.is_dir():
        raise MegaError(ERRORS["assets_missing"])
    python = resolve(config["environment_python"]) if config["environment_python"] else Path(sys.executable)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise MegaError(ERRORS["dependency"])
    started = time.monotonic()
    deadline = started + config["timeout_sec"]
    with tempfile.TemporaryDirectory(prefix="mega-asr-job-", dir=storage) as job:
        directory = Path(job)
        source = directory / "input.audio"
        digest, size = snapshot(path, allowed, source, config["max_file_bytes"], correlation.get("bytes"), deadline, cancel)
        audio = decode_audio(source, directory, config["max_duration_sec"], deadline, cancel)
        with wave.open(str(directory / "input.wav"), "wb") as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000); output.writeframes(audio)
        duration = len(audio) / 32000
        del audio
        job_config = {"assets": str(assets), "job": str(directory), "config": config}
        raw_result = run_process([str(python), "-E", "-B", str(Path(__file__).with_name("worker.py"))],
            directory=directory, deadline=deadline, cancel=cancel, limit=1048576,
            stdin=json.dumps(job_config, ensure_ascii=True).encode())
        result = validate_result(raw_result, duration, config)
        from .assets_store import manifest
        info = manifest()
        result.update(duration_sec=duration, elapsed_sec=round(time.monotonic() - started, 4),
            source={"sha256": digest, "bytes": size, **correlation}, model=info["model"],
            model_revision=info["model_revision"], engine_revision=info["engine_revision"], backend="transformers",
            settings={k: config[k] for k in ("device", "precision", "language", "routing", "quality_threshold",
                "threads", "chunk_duration_sec", "max_new_tokens")},
            timestamps_available=False, final=True)
        return result
