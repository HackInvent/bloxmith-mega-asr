"""Explicitly fetch pinned MEGA-ASR engine/weights into an offline asset directory."""

import argparse
from pathlib import Path
import sys
import types

package = types.ModuleType("mega_asr_setup")
package.__path__ = [str(Path(__file__).resolve().parent)]
sys.modules[package.__name__] = package
from mega_asr_setup.assets_store import AssetError, download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Destination for about 4.8 GB of pinned public assets; never a blueprint edit")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.timeout_sec <= 7200:
        parser.error("timeout-sec must be between 1 and 7200")
    try:
        result = download(args.directory, timeout_sec=args.timeout_sec, progress=lambda name: print("[download] " + name, flush=True))
    except (AssetError, OSError) as exc:
        print("[failed] " + (str(exc) if isinstance(exc, AssetError) else "asset_storage"), file=sys.stderr)
        return 1
    print("[ok] Verified model and engine: " + str(result))
    print("Set asset_directory to this path; install requirements separately in a dedicated Python environment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
