"""
Tests for the Forge preload hook.

Forge hands ``preload`` its argparse parser before the UI loads; the hook must
register the three ASPIRE flags with the documented defaults.
"""

import argparse

import integrations.forge as forge_pkg
from integrations.forge.preload import preload


class RecordingParser:
    """Fake Forge parser that records add_argument calls."""

    def __init__(self):
        self.calls = []

    def add_argument(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestPreload:
    """Tests for preload()."""

    def test_registers_three_flags(self):
        parser = RecordingParser()
        preload(parser)

        names = [args[0] for args, _ in parser.calls]
        assert names == ["--aspire-teacher", "--aspire-critic-path", "--aspire-cache-dir"]

    def test_flag_defaults_and_types(self):
        parser = RecordingParser()
        preload(parser)
        by_name = {args[0]: kwargs for args, kwargs in parser.calls}

        assert by_name["--aspire-teacher"]["default"] == "claude"
        assert by_name["--aspire-critic-path"]["default"] is None
        assert by_name["--aspire-cache-dir"]["default"] is None
        for kwargs in by_name.values():
            assert kwargs["type"] is str
            assert kwargs["help"]

    def test_teacher_help_lists_supported_teachers(self):
        parser = RecordingParser()
        preload(parser)
        help_text = parser.calls[0][1]["help"]

        for teacher in ("claude", "gpt4v", "local"):
            assert teacher in help_text

    def test_works_with_real_argparse_defaults(self):
        parser = argparse.ArgumentParser()
        preload(parser)
        args = parser.parse_args([])

        assert args.aspire_teacher == "claude"
        assert args.aspire_critic_path is None
        assert args.aspire_cache_dir is None

    def test_works_with_real_argparse_overrides(self):
        parser = argparse.ArgumentParser()
        preload(parser)
        args = parser.parse_args(
            [
                "--aspire-teacher", "gpt4v",
                "--aspire-critic-path", "models/critic.pt",
                "--aspire-cache-dir", "cache/dialogues",
            ]
        )

        assert args.aspire_teacher == "gpt4v"
        assert args.aspire_critic_path == "models/critic.pt"
        assert args.aspire_cache_dir == "cache/dialogues"

    def test_returns_none(self):
        assert preload(RecordingParser()) is None


def test_package_docstring_describes_forge_integration():
    assert "Forge" in forge_pkg.__doc__
    assert "aesthetic" in forge_pkg.__doc__
