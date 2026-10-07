"""Refuse a pod whose GPU cannot run this experiment, before anything is installed or downloaded.

`pip install vllm` brings a PyTorch built for CUDA 13, which needs a driver that supports
CUDA 13.0 or later. Running two composite ASPIRE runs side by side needs about 80 GB of GPU
memory. A rented pod can land on a host that has neither (2026-10-07: an A100 80 GB with a
CUDA 12.8 driver failed after setup and a merge had already cost about $0.40). This check takes
seconds and exits 2 with the reason.

Usage: python host_check.py [--min-cuda 13.0] [--min-memory-gb 90]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


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


def main() -> None:  # pragma: no cover - runs on the pod
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-cuda", default="13.0")
    parser.add_argument("--min-memory-gb", type=float, default=0.0)
    args = parser.parse_args()
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
