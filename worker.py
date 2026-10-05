"""Owned offline inference process; imports the checksum-pinned official MEGA-ASR engine."""

from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import resource
import sys
import types
import warnings
import wave

if __package__:
    from .assets_store import AssetError, prepare
    from .logic import configuration
else:
    package = types.ModuleType("mega_asr_owned")
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[package.__name__] = package
    from mega_asr_owned.assets_store import AssetError, prepare
    from mega_asr_owned.logic import configuration


def guard_generation(model, eos_ids, maximum):
    """Never label a token-budget-truncated decode as a complete transcription."""
    original = model.generate

    def generate(*args, **kwargs):
        result = original(*args, **kwargs)
        sequences = result.sequences
        input_ids = kwargs.get("input_ids")
        if input_ids is None or len(sequences.shape) != 2 or sequences.shape[0] != 1:
            raise AssetError("model_failed")
        count = sequences.shape[1] - input_ids.shape[1]
        if count < 1 or count > maximum or int(sequences[0, -1]) not in eos_ids:
            raise AssetError("token_limit")
        return result

    model.generate = generate


def infer(item):
    cfg = configuration(item["config"])
    limit = cfg["memory_limit_mib"] * 1048576
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cfg["timeout_sec"] * cfg["threads"], cfg["timeout_sec"] * cfg["threads"] + 1))
    from importlib.metadata import version
    for name, expected in (("torch", "2.10.0"), ("torchaudio", "2.10.0"), ("qwen-asr", "0.0.6"),
        ("transformers", "4.57.6"), ("safetensors", "0.7.0"), ("scipy", "1.17.1"), ("soundfile", "0.13.1")):
        if version(name).split("+")[0] != expected:
            raise AssetError("dependency")
    import torch
    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)
    if cfg["device"] == "cuda:0" and (not torch.cuda.is_available() or
        (cfg["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported())):
        raise AssetError("device_unavailable")
    job = Path(item["job"])
    weights, engine = prepare(item["assets"], job)
    # Only copied, verified modules can enter these namespaces; no unexpected __init__.py.
    for name, location in (("MegaASR", engine / "MegaASR"), ("MegaASR.model", engine / "MegaASR" / "model")):
        module = types.ModuleType(name)
        module.__path__ = [str(location)]
        sys.modules[name] = module
    from MegaASR.model.megaASR import MegaASR
    warnings.filterwarnings("error", message="LoRA adapter .*missing .*modules.*")
    model = MegaASR(
        model_path=weights / "Qwen3-ASR-1.7B", lora_dir=weights / "mega-asr-merged",
        router_checkpoint=weights / "audio_quality_router" / "best_acc_model.safetensors",
        routing_enabled=cfg["routing"] == "auto", quality_threshold=cfg["quality_threshold"],
        device_map=cfg["device"], quality_device=cfg["device"],
        dtype=getattr(torch, cfg["precision"]), max_inference_batch_size=1,
        max_new_tokens=cfg["max_new_tokens"], keep_delta_on_gpu=False,
        backend="transformers", local_files_only=True, trust_remote_code=False,
    )
    if model.lora_switch is None or not model.lora_switch.items:
        raise AssetError("model_failed")
    eos_ids = json.loads((weights / "Qwen3-ASR-1.7B" / "generation_config.json").read_text())["eos_token_id"]
    guard_generation(model.asr.model.model, eos_ids, cfg["max_new_tokens"])
    chunks, texts, offset = [], [], 0
    with wave.open(str(job / "input.wav"), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise AssetError("model_failed")
        while audio := source.readframes(16000 * cfg["chunk_duration_sec"]):
            chunk_path = job / "chunk.wav"
            with wave.open(str(chunk_path), "wb") as output:
                output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
                output.writeframes(audio)
            with torch.inference_mode():
                result = model.infer(chunk_path, language=None if cfg["language"] == "auto" else cfg["language"],
                    context=cfg["context"], return_objects=True, return_route=True, return_time_stamps=False)
            objects = result["text"]
            if not isinstance(objects, list) or len(objects) != 1:
                raise AssetError("model_failed")
            text, language = objects[0].text, objects[0].language
            if not isinstance(text, str) or len(text) > 32768 or not isinstance(language, str) or len(language) > 128:
                raise AssetError("model_failed")
            text = text.strip()
            end = offset + len(audio) / 32000
            chunks.append({"start_sec": offset, "end_sec": end, "text": text, "language": language,
                "use_lora": result["use_lora"], "degraded_probability": result["degraded_prob"],
                "route_source": result["route_source"]})
            offset = end
            if text:
                texts.append(text)
            if sum(len(t.encode("utf-8")) for t in texts) > 131072:
                raise AssetError("model_failed")
    return {"text": "\n".join(texts), "chunks": chunks}


def main():
    data = sys.stdin.buffer.read(32769)
    if len(data) > 32768:
        raise AssetError("model_failed")
    item = json.loads(data)
    # Third-party diagnostics stay on the bounded stderr pipe, never mixed into JSON.
    with redirect_stdout(sys.stderr):
        result = infer(item)
    sys.stdout.write(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        from importlib.metadata import PackageNotFoundError
        if isinstance(exc, (ImportError, PackageNotFoundError)):
            code = "dependency"
        elif isinstance(exc, AssetError):
            code = str(exc)
        elif isinstance(exc, MemoryError) or isinstance(exc, RuntimeError) and any(
            marker in str(exc) for marker in ("DefaultCPUAllocator", "Cannot allocate memory", "out of memory")):
            code = "resource_limit"
        else:
            code = "model_failed"
        print(json.dumps({"error_code": code}))
        raise SystemExit(1) from None
