"""Pinned upstream assets. No model SDK import, token lookup or implicit download."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class AssetError(ValueError):
    pass


def manifest():
    return json.loads(Path(__file__).with_name("model_artifacts.json").read_text(encoding="utf-8"))


def safe_path(root, relative):
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise AssetError("assets_missing")
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise AssetError("assets_corrupt")
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise AssetError("assets_corrupt")
    return candidate


def verify(root, spec):
    target = safe_path(root, spec["path"])
    try:
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    except OSError:
        raise AssetError("assets_corrupt") from None
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != spec["bytes"]:
            raise AssetError("assets_corrupt")
        digest = hashlib.sha256()
        while chunk := stream.read(1048576):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        if digest.hexdigest() != spec["sha256"] or (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise AssetError("assets_corrupt")
    return True


class PinnedRedirects(HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or parsed.username or parsed.password or not (
            host in {"raw.githubusercontent.com", "huggingface.co"} or host.endswith(".huggingface.co") or host.endswith(".hf.co")):
            raise AssetError("download_failed")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(root, *, timeout_sec=1800, progress=lambda item: None):
    """Explicit setup only. Verify every byte; never replace an existing corrupt asset."""
    root = Path(root).absolute()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    specs = manifest()["artifacts"]
    missing = [spec for spec in specs if not verify(root, spec)]
    if shutil.disk_usage(root).free < sum(spec["bytes"] for spec in missing) + 1073741824:
        raise AssetError("disk_space")
    deadline = time.monotonic() + timeout_sec
    opener = build_opener(ProxyHandler({}), PinnedRedirects())
    for spec in missing:
        progress(spec["path"])
        target = safe_path(root, spec["path"])
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mega-download-", dir=root) as job:
            temporary = Path(job) / "asset"
            request = Request(spec["url"], headers={"User-Agent": "BloxSmith-MEGA-ASR/0.1.0", "Accept-Encoding": "identity"})
            try:
                with opener.open(request, timeout=20) as source, open(temporary, "xb") as output:
                    os.chmod(temporary, 0o600)
                    digest, total = hashlib.sha256(), 0
                    while True:
                        if time.monotonic() >= deadline:
                            raise AssetError("download_timeout")
                        chunk = source.read(min(1048576, spec["bytes"] + 1 - total))
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > spec["bytes"]:
                            raise AssetError("download_size")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if total != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
                    raise AssetError("download_integrity")
                try:
                    os.link(temporary, target, follow_symlinks=False)
                except FileExistsError:
                    if not verify(root, spec):
                        raise AssetError("assets_corrupt") from None
            except AssetError:
                raise
            except Exception:
                raise AssetError("download_failed") from None
    return root


def prepare(root, job):
    """Verify immutable weights and snapshot only the declared upstream Python files."""
    info = manifest()
    root, job = Path(root), Path(job)
    for spec in info["artifacts"]:
        if not verify(root, spec):
            raise AssetError("assets_missing")
        if spec["path"].startswith("engine/src/"):
            target = job / spec["path"]
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            data = safe_path(root, spec["path"]).read_bytes()
            if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise AssetError("assets_corrupt")
            with target.open("xb") as output:
                output.write(data)
    return root / "weights", job / "engine" / "src"
