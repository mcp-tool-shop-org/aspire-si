"""Refuse a pod whose GPU cannot run this experiment, before anything is installed or downloaded.

`pip install vllm` brings a PyTorch built for CUDA 13, which needs a driver that supports
CUDA 13.0 or later. Running two composite ASPIRE runs side by side needs about 80 GB of GPU
memory. A rented pod can land on a host that has neither (2026-10-07: an A100 80 GB with a
CUDA 12.8 driver failed after setup and a merge had already cost about $0.40). This check takes
seconds and exits 2 with the reason.

With --speed it checks something else, after the installs, in about 30 s: whether the host can
fetch the ~130 GB of teachers in time (2026-10-07: a pod whose /workspace was a network
filesystem took 31 minutes for 26 GB, so seed 43 could not finish inside its cap and was stopped
after $1.65). It measures the two things that were slow there:

  writes   512 MiB written to the Hugging Face cache directory and flushed, against
           --min-write-mbps (100). That pod's network filesystem wrote 32 MB/s.
  network  one plain HTTPS stream of a model file for up to --stream-seconds (20), against
           --min-stream-mbps (20). That pod got 1.9 MB/s; a home connection gets about 46.

The network probe is plain HTTP on purpose: the huggingface_hub download path measured 2 MB/s on
a connection where one HTTP stream ran at 46 MB/s, so it would judge the library, not the host.

Usage: python host_check.py [--min-cuda 13.0] [--min-memory-gb 90]
       python host_check.py --speed [--url URL] [--min-write-mbps 100] [--min-stream-mbps 20]
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


DEFAULT_URL = "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/resolve/main/model-00001-of-00017.safetensors"


def stream_rate(url: str, seconds: float, chunk: int = 1 << 20) -> tuple[int, float]:  # pragma: no cover
    """Bytes read from one HTTPS stream of `url`, and the time taken, stopping after `seconds`."""
    import urllib.request

    started = time.monotonic()
    nbytes = 0
    with urllib.request.urlopen(url, timeout=30) as response:
        while time.monotonic() - started < seconds:
            data = response.read(chunk)
            if not data:
                break
            nbytes += len(data)
    return nbytes, time.monotonic() - started


def check_speed(
    url: str, min_write_mbps: float, min_stream_mbps: float, stream_seconds: float
) -> None:  # pragma: no cover
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    found = []
    nbytes, seconds = write_rate(cache, 512)
    print(f"write: {nbytes / 1e9:.2f} GB to {cache} in {seconds:.1f} s")
    found.append(rate_problem(f"writes to {cache}", nbytes, seconds, min_write_mbps))
    nbytes, seconds = stream_rate(url, stream_seconds)
    print(f"network: {nbytes / 1e9:.2f} GB in {seconds:.1f} s from {url.split('/resolve/')[0]}")
    found.append(rate_problem("one download stream", nbytes, seconds, min_stream_mbps))
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
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--min-write-mbps", type=float, default=100.0)
    parser.add_argument("--min-stream-mbps", type=float, default=20.0)
    parser.add_argument("--stream-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.speed:
        check_speed(args.url, args.min_write_mbps, args.min_stream_mbps, args.stream_seconds)
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
