"""Loading a checkpoint must never run code from it.

`aspire judge` and the trainers open `.pt` files a user may have downloaded. With
`weights_only=False` (torch's default before 2.6, and aspire-si allows torch 2.0), `torch.load`
unpickles arbitrary objects, so a crafted file runs code. Every load passes `weights_only=True`.
"""

import pickle
from pathlib import Path

import pytest
import torch

from aspire.critic import CriticHead

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "aspire", ROOT / "integrations"]

RAN = []


class _Payload:
    """Unpickling this calls RAN.append: stands in for any code a crafted file would run."""

    def __reduce__(self):
        return (RAN.append, ("ran",))


def _crafted(path: Path) -> Path:
    torch.save({"state_dict": {}, "config": {"input_dim": 8}, "payload": _Payload()}, path)
    return path


def test_every_torch_load_passes_weights_only():
    calls = []
    for source in SOURCES:
        for file in source.rglob("*.py"):
            for number, line in enumerate(file.read_text(encoding="utf8").splitlines(), 1):
                if "torch.load(" in line and not line.lstrip().startswith("#"):
                    calls.append((file.relative_to(ROOT).as_posix(), number, line.strip()))
    assert calls, "expected at least one torch.load call"
    unsafe = [call for call in calls if "weights_only=True" not in call[2]]
    assert unsafe == []


def test_critic_checkpoint_round_trips_under_weights_only(tmp_path):
    critic = CriticHead(input_dim=8, hidden_dim=16, reasoning_dim=12).eval()
    path = tmp_path / "critic.pt"
    critic.save(str(path))
    loaded = CriticHead.load(str(path)).eval()
    for a, b in zip(critic.parameters(), loaded.parameters()):
        assert torch.equal(a, b)


def test_crafted_critic_checkpoint_is_refused_without_running(tmp_path):
    RAN.clear()
    path = _crafted(tmp_path / "critic.pt")
    with pytest.raises(pickle.UnpicklingError):
        CriticHead.load(str(path))
    assert RAN == []
