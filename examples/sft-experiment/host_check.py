"""Refuse a pod whose GPU cannot run this experiment, before anything is installed or downloaded.

`pip install vllm` brings a PyTorch built for CUDA 13, which needs a driver that supports
CUDA 13.0 or later. Running two composite ASPIRE runs side by side needs about 80 GB of GPU
memory. A rented pod can land on a host that has neither (2026-10-07: an A100 80 GB with a
CUDA 12.8 driver failed after setup and a merge had already cost about $0.40). This check takes
seconds and exits 2 with the reason.

With --speed it checks something else, after the installs: whether the host can fetch the
teacher in the time the plan allows. It measures:

  writes    512 MiB written to the Hugging Face cache directory and flushed, against
            --min-write-mbps (100). (2026-10-07: a pod whose /workspace was a network filesystem
            wrote 32 MB/s, and a seed run could not finish inside its cap.)
  download  one shard of the teacher (--model, --shard), fetched through huggingface_hub into the
            cache the run uses, exactly as the run's own download goes. The floor is not a fixed
            number: it is --download-gb in --download-minutes, the time the plan's budget gives
            the download. The shard stays in the cache, so the run does not fetch it again.

The download probe used to be one plain HTTPS stream. On 2026-10-08 that stream measured 20 MB/s
and passed, while the run's own hf download averaged about 11 MB/s, so the 32B teacher could not
arrive in time and the run was stopped ($1.86). The probe now measures the path the run takes.
Set HF_XET_HIGH_PERFORMANCE=1 (and any other download settings) before the probe, so it measures
them too.

Usage: python host_check.py [--min-cuda 13.0] [--min-memory-gb 90]
       python host_check.py --speed --download-gb 66 --download-minutes 45
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def driver_cuda(smi_output: str) -> tuple[int, int] | None:
    """The highest CUDA version the driver supports, from `nvidia-smi`'s header.

    Older drivers print "CUDA Version: 12.8"; newer ones print "CUDA UMD Version: 13.4".
    """
    match = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", smi_output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def gpus(query_output: str) -> list[tuple[str, float]]:
    """(name, memory in GB) per GPU, from `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`."""
    rows = []
    for line in query_output.strip().splitlines():
        name, _, memory = line.rpartition(",")
        mib = re.search(r"(\d+)", memory)
        if name and mib:
            rows.append((name.strip(), int(mib.group(1)) / 1024))
    return rows


def problems(
    smi_output: str, query_output: str, min_cuda: tuple[int, int], min_memory_gb: float
) -> list[str]:
    found = []
    cuda = driver_cuda(smi_output)
    if cuda is None:
        found.append("nvidia-smi reported no CUDA version")
    elif cuda < min_cuda:
        found.append(
            f"the driver supports CUDA {cuda[0]}.{cuda[1]}; this experiment needs {min_cuda[0]}.{min_cuda[1]}"
        )
    cards = gpus(query_output)
    if not cards:
        found.append("nvidia-smi reported no GPU")
    else:
        name, gb = max(cards, key=lambda card: card[1])
        if gb < min_memory_gb:
            found.append(f"the largest GPU ({name}) has {gb:.0f} GB; this needs {min_memory_gb:.0f} GB")
    return found


def rate_problem(what: str, nbytes: int, seconds: float, min_mbps: float) -> str | None:
    """Why a measured transfer is too slow, or None. MB/s here is 10^6 bytes per second."""
    rate = nbytes / 1e6 / max(seconds, 1e-6)
    if rate < min_mbps:
        measured = f"{nbytes / 1e9:.2f} GB in {seconds:.0f} s"
        return f"{what} ran at {rate:.0f} MB/s ({measured}); this needs {min_mbps:.0f} MB/s"
    return None


def write_rate(directory: Path, megabytes: int = 1024) -> tuple[int, float]:
    """Bytes written and seconds taken for one file of `megabytes` MiB, flushed to disk, in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".host_check_write_probe"
    block = os.urandom(1 << 20)
    started = time.monotonic()
    try:
        with open(probe, "wb", buffering=0) as f:
            for _ in range(megabytes):
                f.write(block)
            os.fsync(f.fileno())
        seconds = time.monotonic() - started
    finally:
        probe.unlink(missing_ok=True)
    return megabytes << 20, seconds


DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_SHARD = "model-00001-of-00017.safetensors"


def required_mbps(download_gb: float, download_minutes: float) -> float:
    """The download rate, in MB/s, that fetches `download_gb` in `download_minutes`."""
    return download_gb * 1000 / (download_minutes * 60)


def projected_minutes(download_gb: float, nbytes: int, seconds: float) -> float:
    """How long `download_gb` takes at the rate measured on the shard."""
    rate = nbytes / 1e6 / max(seconds, 1e-6)
    return download_gb * 1000 / max(rate, 1e-6) / 60


def probe_timeout(shard_bytes: int, need_mbps: float) -> float:
    """Seconds to wait for the shard: twice what the floor allows for it. A host that takes
    longer is refused without waiting for the whole shard."""
    return 2 * shard_bytes / 1e6 / need_mbps


def shard_rate(model: str, shard: str, need_mbps: float) -> tuple[int, float]:  # pragma: no cover
    """Bytes and seconds for one shard fetched through huggingface_hub into the run's cache.
    Refuses the host (exit 2) if the shard takes longer than `probe_timeout`."""
    import threading

    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

    size = get_hf_file_metadata(hf_hub_url(model, shard)).size or 0
    limit = probe_timeout(size, need_mbps)
    done: list[str] = []
    started = time.monotonic()
    worker = threading.Thread(target=lambda: done.append(hf_hub_download(model, shard)), daemon=True)
    worker.start()
    worker.join(limit)
    seconds = time.monotonic() - started
    if not done:
        print(
            f"HOST-REFUSED: {shard} of {model} ({size / 1e9:.2f} GB) did not arrive in {limit:.0f} s, "
            f"twice what {need_mbps:.0f} MB/s allows",
            flush=True,
        )
        os._exit(2)
    return Path(done[0]).stat().st_size, seconds


def check_speed(
    model: str, shard: str, min_write_mbps: float, download_gb: float, download_minutes: float
) -> None:  # pragma: no cover
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    found = []
    nbytes, seconds = write_rate(cache, 512)
    print(f"write: {nbytes / 1e9:.2f} GB to {cache} in {seconds:.1f} s")
    found.append(rate_problem(f"writes to {cache}", nbytes, seconds, min_write_mbps))
    settings = {k: v for k, v in os.environ.items() if k.startswith(("HF_XET", "HF_HUB"))}
    need = required_mbps(download_gb, download_minutes)
    nbytes, seconds = shard_rate(model, shard, need)
    eta = projected_minutes(download_gb, nbytes, seconds)
    print(
        f"download: {shard} of {model}, {nbytes / 1e9:.2f} GB in {seconds:.0f} s; "
        f"{download_gb:.0f} GB would take {eta:.0f} min (budget {download_minutes:.0f}); settings {settings}"
    )
    found.append(rate_problem(f"the {model} download", nbytes, seconds, need))
    found = [f for f in found if f]
    if found:
        print("HOST-REFUSED: " + "; ".join(found), flush=True)
        sys.exit(2)
    print("SPEED-OK")


def main() -> None:  # pragma: no cover - runs on the pod
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-cuda", default="13.0")
    parser.add_argument("--min-memory-gb", type=float, default=0.0)
    parser.add_argument("--speed", action="store_true", help="check write and download speed, not the GPU")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the model the run downloads")
    parser.add_argument("--shard", default=DEFAULT_SHARD, help="one of its files, timed")
    parser.add_argument("--min-write-mbps", type=float, default=100.0)
    parser.add_argument("--download-gb", type=float, default=66.0, help="everything the run downloads")
    parser.add_argument("--download-minutes", type=float, default=45.0, help="the plan's time for it")
    args = parser.parse_args()
    if args.speed:
        check_speed(args.model, args.shard, args.min_write_mbps, args.download_gb, args.download_minutes)
        return
    major, minor = (int(x) for x in args.min_cuda.split("."))
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    found = problems(smi, query, (major, minor), args.min_memory_gb)
    print(
        "host:",
        ", ".join(f"{n} {gb:.0f} GB" for n, gb in gpus(query)) or "no GPU",
        f"| CUDA {driver_cuda(smi)}",
    )
    if found:
        print("HOST-REFUSED: " + "; ".join(found), flush=True)
        sys.exit(2)
    print("HOST-OK")


if __name__ == "__main__":
    main()
