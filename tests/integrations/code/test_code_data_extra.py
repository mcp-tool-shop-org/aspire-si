"""
Extra tests for integrations/code/data.py: collection edge cases, dataset
content, balanced sampling and streaming. Everything is local and offline;
git is never invoked.
"""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from integrations.code.code_teacher import CodeCritique, CodeTeacher
from integrations.code.config import Language
from integrations.code.data import (
    CodeReviewDataset,
    CodeReviewPair,
    GitHubRepoCollector,
    StreamingCodeDataset,
    create_balanced_dataset,
    generate_training_pairs,
)

from .conftest import FakeTokenizer


def make_critique(score=5.0, **kw) -> CodeCritique:
    return CodeCritique(overall_score=score, teacher_name="T", language=Language.PYTHON, **kw)


def make_pair(code="x = 1", score=5.0, **kw) -> CodeReviewPair:
    return CodeReviewPair(
        code=code,
        language=Language.PYTHON,
        critique=make_critique(score, **kw.pop("critique_kw", {})),
        **kw,
    )


@pytest.fixture
def collector(tmp_path):
    return GitHubRepoCollector(cache_dir=str(tmp_path / "cache"))


@pytest.fixture
def repo_dir():
    """A repo directory whose absolute path avoids collect_files' skip words ('test', 'example', ...)."""
    path = Path(tempfile.mkdtemp(prefix="dcol_"))
    if any(w in str(path).lower() for w in ("test", "example", "vendor", "node_modules")):
        shutil.rmtree(path, ignore_errors=True)
        pytest.skip("system temp directory contains a collect_files skip word")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def write_py(path: Path, lines: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * lines)


# ============================================================================
# CodeReviewPair
# ============================================================================


class TestCodeReviewPairSerialisation:
    def test_to_dict_exact_keys(self):
        pair = make_pair(
            "code", 6.5, filename="a.py", repo="o/r", commit="abc123", improved_code="better",
            critique_kw={"reasoning": "why", "strengths": ["s"], "weaknesses": ["w"], "suggestions": ["g"]},
        )
        assert pair.to_dict() == {
            "code": "code", "language": "python", "score": 6.5, "reasoning": "why",
            "strengths": ["s"], "weaknesses": ["w"], "suggestions": ["g"],
            "filename": "a.py", "repo": "o/r", "commit": "abc123", "improved_code": "better",
        }

    def test_from_dict_applies_defaults(self):
        pair = CodeReviewPair.from_dict({"code": "c", "language": "rust", "score": 3})
        assert pair.language == Language.RUST
        assert pair.critique.overall_score == 3
        assert pair.critique.reasoning == ""
        assert pair.critique.strengths == [] and pair.critique.weaknesses == [] and pair.critique.suggestions == []
        assert pair.critique.teacher_name == "loaded"
        assert pair.critique.language == Language.RUST
        assert (pair.filename, pair.repo, pair.improved_code) == (None, None, None)

    def test_from_dict_missing_required_key(self):
        with pytest.raises(KeyError, match="score"):
            CodeReviewPair.from_dict({"code": "c", "language": "python"})

    def test_from_dict_invalid_language(self):
        with pytest.raises(ValueError):
            CodeReviewPair.from_dict({"code": "c", "language": "cobol", "score": 1})

    def test_commit_survives_round_trip(self):
        pair = make_pair(commit="abc123")
        assert CodeReviewPair.from_dict(pair.to_dict()).commit == "abc123"


# ============================================================================
# GitHubRepoCollector
# ============================================================================


class TestCollectorInit:
    def test_cache_dir_created_recursively(self, tmp_path):
        c = GitHubRepoCollector(cache_dir=str(tmp_path / "a" / "b" / "c"))
        assert c.cache_dir.is_dir()

    def test_tilde_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        c = GitHubRepoCollector(cache_dir="~/repos_cache")
        assert c.cache_dir == tmp_path / "repos_cache"
        assert c.cache_dir.is_dir()

    def test_token_from_argument_wins_over_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "from-env")
        assert GitHubRepoCollector(str(tmp_path), github_token="explicit").github_token == "explicit"

    def test_token_falls_back_to_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "from-env")
        assert GitHubRepoCollector(str(tmp_path)).github_token == "from-env"

    def test_token_none_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        assert GitHubRepoCollector(str(tmp_path)).github_token is None


class TestCloneRepo:
    def test_full_clone_has_no_depth_flag(self, collector):
        with patch("integrations.code.data.subprocess.run") as run:
            path = collector.clone_repo("owner/name", shallow=False)
        cmd = run.call_args.args[0]
        assert cmd == ["git", "clone", "https://github.com/owner/name.git", str(path)]
        assert run.call_args.kwargs["check"] is True

    def test_shallow_clone_flags_and_target_dir(self, collector):
        with patch("integrations.code.data.subprocess.run") as run:
            path = collector.clone_repo("owner/name")
        assert run.call_args.args[0][:4] == ["git", "clone", "--depth", "1"]
        assert path == collector.cache_dir / "owner_name"

    def test_clone_error_includes_repo_and_stderr(self, collector):
        err = subprocess.CalledProcessError(128, "git", stderr=b"fatal: repo not found")
        with patch("integrations.code.data.subprocess.run", side_effect=err):
            with pytest.raises(RuntimeError, match="Failed to clone owner/name: fatal: repo not found"):
                collector.clone_repo("owner/name")


class TestCollectFiles:
    def test_yields_relative_paths_and_code(self, collector, repo_dir):
        write_py(repo_dir / "pkg" / "mod.py", 20)
        with patch.object(collector, "clone_repo", return_value=repo_dir):
            files = list(collector.collect_files("o/r", Language.PYTHON))
        assert files == [(str(Path("pkg") / "mod.py"), "x = 1\n" * 20)]

    def test_unsupported_language_yields_nothing(self, collector, repo_dir):
        write_py(repo_dir / "a.py")
        with patch.object(collector, "clone_repo", return_value=repo_dir):
            assert list(collector.collect_files("o/r", Language.RUBY)) == []

    def test_max_files_caps_output(self, collector, repo_dir):
        for i in range(5):
            write_py(repo_dir / f"m{i}.py")
        with patch.object(collector, "clone_repo", return_value=repo_dir):
            assert len(list(collector.collect_files("o/r", Language.PYTHON, max_files=3))) == 3
            assert list(collector.collect_files("o/r", Language.PYTHON, max_files=0)) == []

    def test_max_files_applies_across_extensions(self, collector, repo_dir):
        (repo_dir / "a.js").write_text("a\n" * 20)
        (repo_dir / "b.mjs").write_text("b\n" * 20)
        with patch.object(collector, "clone_repo", return_value=repo_dir):
            names = [n for n, _ in collector.collect_files("o/r", Language.JAVASCRIPT, max_files=10)]
            capped = list(collector.collect_files("o/r", Language.JAVASCRIPT, max_files=1))
        assert sorted(names) == ["a.js", "b.mjs"]
        assert len(capped) == 1

    @pytest.mark.parametrize("skipped", ["tests", "examples", "vendor", "node_modules", "__pycache__"])
    def test_skip_directories(self, collector, repo_dir, skipped):
        write_py(repo_dir / skipped / "m.py")
        write_py(repo_dir / "src" / "keep.py")
        with patch.object(collector, "clone_repo", return_value=repo_dir):
            names = [n for n, _ in collector.collect_files("o/r", Language.PYTHON)]
        assert names == [str(Path("src") / "keep.py")]

    def test_line_limits_are_inclusive(self, collector, repo_dir):
        write_py(repo_dir / "low.py", 9)     # 10 lines after split (trailing newline)
        write_py(repo_dir / "high.py", 499)  # 500 lines after split
        write_py(repo_dir / "over.py", 500)  # 501 lines
        write_py(repo_dir / "under.py", 8)   # 9 lines
        with patch.object(collector, "clone_repo", return_value=repo_dir):
            names = sorted(n for n, _ in collector.collect_files("o/r", Language.PYTHON))
        assert names == ["high.py", "low.py"]

    def test_unreadable_file_is_skipped_with_debug_log(self, collector, repo_dir, caplog):
        write_py(repo_dir / "bad.py")
        write_py(repo_dir / "good.py")
        real_read = Path.read_text

        def flaky(self, *a, **kw):
            if self.name == "bad.py":
                raise OSError("permission denied")
            return real_read(self, *a, **kw)

        with patch.object(Path, "read_text", flaky), patch.object(collector, "clone_repo", return_value=repo_dir):
            with caplog.at_level(logging.DEBUG, logger="integrations.code.data"):
                names = [n for n, _ in collector.collect_files("o/r", Language.PYTHON)]
        assert names == ["good.py"]
        assert "permission denied" in caplog.text

    def test_undecodable_bytes_are_replaced_not_fatal(self, collector, repo_dir):
        (repo_dir / "bin.py").write_bytes(b"x = '\xff\xfe'\n" * 20)
        with patch.object(collector, "clone_repo", return_value=repo_dir):
            ((name, code),) = collector.collect_files("o/r", Language.PYTHON)
        assert name == "bin.py" and code.startswith("x = '")

    def test_custom_line_bounds(self, collector, repo_dir):
        write_py(repo_dir / "five.py", 4)
        with patch.object(collector, "clone_repo", return_value=repo_dir):
            assert len(list(collector.collect_files("o/r", Language.PYTHON, min_lines=1, max_lines=5))) == 1
            assert list(collector.collect_files("o/r", Language.PYTHON, min_lines=6)) == []

    def test_repo_location_does_not_affect_collection(self, collector, tmp_path):
        repo = tmp_path / "latest_checkout" / "repo"  # tmp_path itself contains the test's name
        write_py(repo / "src" / "keep.py")
        with patch.object(collector, "clone_repo", return_value=repo):
            names = [n for n, _ in collector.collect_files("o/r", Language.PYTHON)]
        assert names == [str(Path("src") / "keep.py")]


class TestCollectFromQualityRepos:
    def test_yields_repo_filename_code_for_each_repo(self, collector):
        calls = []

        def fake_collect(repo, language, max_files):
            calls.append((repo, language, max_files))
            yield "f.py", f"code-of-{repo}"

        with patch.object(collector, "collect_files", side_effect=fake_collect):
            out = list(collector.collect_from_quality_repos(Language.TYPESCRIPT, files_per_repo=7))
        repos = collector.QUALITY_REPOS[Language.TYPESCRIPT]
        assert out == [(r, "f.py", f"code-of-{r}") for r in repos]
        assert calls == [(r, Language.TYPESCRIPT, 7) for r in repos]

    def test_language_without_curated_repos_yields_nothing(self, collector):
        with patch.object(collector, "collect_files") as cf:
            assert list(collector.collect_from_quality_repos(Language.RUBY)) == []
        cf.assert_not_called()

    def test_os_errors_skip_repo_and_log_warning(self, collector, caplog):
        def fake_collect(repo, language, max_files):
            if repo == "microsoft/TypeScript":
                raise OSError("disk full")
            yield "f.ts", repo

        with patch.object(collector, "collect_files", side_effect=fake_collect):
            with caplog.at_level(logging.WARNING, logger="integrations.code.data"):
                out = list(collector.collect_from_quality_repos(Language.TYPESCRIPT))
        assert [r for r, _, _ in out] == ["microsoft/vscode", "angular/angular"]
        assert "Failed to collect from microsoft/TypeScript" in caplog.text

    def test_called_process_error_skips_repo(self, collector):
        def fake_collect(repo, language, max_files):
            if repo == "golang/go":
                raise subprocess.CalledProcessError(1, "git")
            yield "f.go", repo

        with patch.object(collector, "collect_files", side_effect=fake_collect):
            out = list(collector.collect_from_quality_repos(Language.GO))
        assert [r for r, _, _ in out] == ["kubernetes/kubernetes", "docker/docker-ce"]

    def test_failed_clone_does_not_abort_remaining_repos(self, collector):
        def fake_collect(repo, language, max_files):
            if repo == "golang/go":
                raise RuntimeError("Failed to clone golang/go: not found")
            yield "f.go", repo

        with patch.object(collector, "collect_files", side_effect=fake_collect):
            out = list(collector.collect_from_quality_repos(Language.GO))
        assert [r for r, _, _ in out] == ["kubernetes/kubernetes", "docker/docker-ce"]


# ============================================================================
# generate_training_pairs
# ============================================================================


class TestGenerateTrainingPairs:
    def test_pairs_carry_metadata_and_critique(self):
        teacher = MagicMock(spec=CodeTeacher)
        teacher.critique.side_effect = lambda s: make_critique(len(s.code) / 2)
        pairs = generate_training_pairs(
            teacher, [("o/r", "a.py", "1234"), ("o/s", "b.py", "12345678")], Language.PYTHON
        )
        assert [(p.repo, p.filename, p.code, p.language) for p in pairs] == [
            ("o/r", "a.py", "1234", Language.PYTHON), ("o/s", "b.py", "12345678", Language.PYTHON),
        ]
        assert [p.critique.overall_score for p in pairs] == [2.0, 4.0]
        sample = teacher.critique.call_args_list[0].args[0]
        assert (sample.filename, sample.language, sample.code) == ("a.py", Language.PYTHON, "1234")

    def test_failing_critiques_are_dropped_and_logged(self, caplog):
        teacher = MagicMock(spec=CodeTeacher)
        teacher.critique.side_effect = [RuntimeError("api down"), make_critique(7.0)]
        with caplog.at_level(logging.WARNING, logger="integrations.code.data"):
            pairs = generate_training_pairs(
                teacher, [("o/r", "bad.py", "x"), ("o/r", "ok.py", "y")], Language.PYTHON
            )
        assert [p.filename for p in pairs] == ["ok.py"]
        assert "Failed to critique bad.py: api down" in caplog.text

    def test_empty_input(self):
        assert generate_training_pairs(MagicMock(spec=CodeTeacher), [], Language.PYTHON) == []


# ============================================================================
# CodeReviewDataset
# ============================================================================


class RecordingTokenizer(FakeTokenizer):
    def __init__(self):
        super().__init__()
        self.texts: list[str] = []

    def __call__(self, text, **kw):
        self.texts.append(text)
        return super().__call__(text, **kw)


class TestCodeReviewDatasetContent:
    def test_critic_item_contents(self):
        tok = RecordingTokenizer()
        ds = CodeReviewDataset([make_pair("abc", 7.25)], tok, max_length=8, mode="critic")
        item = ds[0]
        assert tok.texts == ["abc"]
        assert item["input_ids"].shape == (8,)
        assert item["attention_mask"].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
        assert item["score"].dtype == torch.float32 and item["score"].item() == 7.25
        assert item["language"] == "python"

    def test_critic_item_truncates_long_code(self):
        ds = CodeReviewDataset([make_pair("z" * 50)], RecordingTokenizer(), max_length=8)
        assert ds[0]["attention_mask"].sum().item() == 8

    def test_student_prompt_lists_first_three_weaknesses(self):
        tok = RecordingTokenizer()
        pair = make_pair("print(1)", 4.0, critique_kw={"weaknesses": ["w1", "w2", "w3", "w4"]})
        CodeReviewDataset([pair], tok, max_length=8, mode="student")[0]
        prompt, target = tok.texts
        assert prompt == (
            "Review this code and improve it:\n\n```python\nprint(1)\n```\n\nIssues: w1; w2; w3\n\nImproved code:"
        )
        assert target == "print(1)"  # no improved_code -> original code is the target

    def test_student_target_prefers_improved_code(self):
        tok = RecordingTokenizer()
        pair = make_pair("old", improved_code="new")
        item = CodeReviewDataset([pair], tok, max_length=4, mode="student")[0]
        assert tok.texts[1] == "new"
        assert item["labels"].shape == (4,)
        assert set(item) == {"input_ids", "attention_mask", "labels", "score"}

    def test_student_prompt_without_weaknesses(self):
        tok = RecordingTokenizer()
        CodeReviewDataset([make_pair("x")], tok, max_length=4, mode="student")[0]
        assert "Issues: \n\nImproved code:" in tok.texts[0]

    def test_index_out_of_range(self):
        ds = CodeReviewDataset([make_pair()], RecordingTokenizer(), max_length=4)
        with pytest.raises(IndexError):
            ds[1]

    def test_works_with_dataloader_batching(self):
        pairs = [make_pair(f"v{i}", float(i)) for i in range(5)]
        ds = CodeReviewDataset(pairs, FakeTokenizer(), max_length=6)
        batches = list(torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0))
        assert [tuple(b["input_ids"].shape) for b in batches] == [(2, 6), (2, 6), (1, 6)]
        assert torch.cat([b["score"] for b in batches]).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


# ============================================================================
# create_balanced_dataset
# ============================================================================


def pairs_with_scores(scores):
    return [make_pair(f"c{i}", s) for i, s in enumerate(scores)]


def bin_counts(pairs, bins=5):
    counts = [0] * bins
    for p in pairs:
        counts[min(int(p.critique.overall_score / (10.0 / bins)), bins - 1)] += 1
    return counts


class TestCreateBalancedDataset:
    def test_each_bin_capped_at_smallest_nonempty_bin(self):
        random.seed(0)
        scores = [0.5] * 6 + [3.0] * 4 + [9.0] * 2  # bins 0, 1, 4 populated
        out = create_balanced_dataset(pairs_with_scores(scores))
        assert bin_counts(out) == [2, 2, 0, 0, 2]

    def test_perfect_ten_lands_in_last_bin(self):
        random.seed(0)
        out = create_balanced_dataset(pairs_with_scores([10.0, 10.0, 9.9]), score_bins=5)
        assert bin_counts(out) == [0, 0, 0, 0, 3]

    def test_explicit_samples_per_bin_larger_than_bin_keeps_all(self):
        random.seed(0)
        out = create_balanced_dataset(pairs_with_scores([1.0, 1.5, 7.0]), samples_per_bin=10)
        assert sorted(p.code for p in out) == ["c0", "c1", "c2"]

    def test_explicit_samples_per_bin_downsamples(self):
        random.seed(0)
        out = create_balanced_dataset(pairs_with_scores([1.0] * 5 + [7.0] * 5), samples_per_bin=2)
        assert bin_counts(out) == [2, 0, 0, 2, 0]

    def test_custom_bin_count(self):
        random.seed(0)
        out = create_balanced_dataset(pairs_with_scores([1.0] * 4 + [6.0] * 2), score_bins=2)
        assert bin_counts(out, 2) == [2, 2]

    def test_returns_subset_of_inputs_without_duplicates(self):
        random.seed(1)
        pairs = pairs_with_scores([1.0] * 3 + [5.0] * 3 + [9.0] * 3)
        out = create_balanced_dataset(pairs)
        assert len(out) == len({id(p) for p in out}) == 9
        assert {id(p) for p in out} <= {id(p) for p in pairs}

    def test_input_list_not_mutated(self):
        pairs = pairs_with_scores([1.0, 5.0, 9.0])
        snapshot = list(pairs)
        create_balanced_dataset(pairs)
        assert pairs == snapshot

    def test_empty_input_returns_empty_list(self):
        assert create_balanced_dataset([]) == []


# ============================================================================
# StreamingCodeDataset
# ============================================================================


class TestStreamingCodeDatasetExtra:
    @pytest.fixture
    def teacher(self):
        t = MagicMock(spec=CodeTeacher)
        t.critique.side_effect = lambda s: make_critique(float(len(s.code)))
        return t

    def test_defaults_to_curated_repos_for_language(self, collector, teacher):
        ds = StreamingCodeDataset(collector, teacher, FakeTokenizer(), Language.GO)
        assert ds.repos == collector.QUALITY_REPOS[Language.GO]

    def test_unknown_language_has_no_repos(self, collector, teacher):
        ds = StreamingCodeDataset(collector, teacher, FakeTokenizer(), Language.RUBY)
        assert ds.repos == [] and list(ds) == []

    def test_explicit_repos_override_defaults(self, collector, teacher):
        ds = StreamingCodeDataset(collector, teacher, FakeTokenizer(), Language.GO, repos=["me/mine"])
        assert ds.repos == ["me/mine"]

    def test_yield_contents_across_repos(self, collector, teacher):
        files = {"a/a": [("x.py", "abc"), ("y.py", "abcdef")], "b/b": [("z.py", "12")]}
        collector.collect_files = MagicMock(side_effect=lambda repo, lang: iter(files[repo]))
        ds = StreamingCodeDataset(
            collector, teacher, FakeTokenizer(), Language.PYTHON, max_length=6, repos=["a/a", "b/b"]
        )
        items = list(ds)
        assert [(i["repo"], i["filename"]) for i in items] == [("a/a", "x.py"), ("a/a", "y.py"), ("b/b", "z.py")]
        assert [i["score"].item() for i in items] == [3.0, 6.0, 2.0]
        assert all(i["input_ids"].shape == (6,) and i["attention_mask"].shape == (6,) for i in items)
        assert items[2]["attention_mask"].tolist() == [1, 1, 0, 0, 0, 0]

    def test_collection_error_skips_repo_and_logs(self, collector, teacher, caplog):
        def fake_collect(repo, lang):
            if repo == "bad/repo":
                raise RuntimeError("clone failed")
            return iter([("ok.py", "abc")])

        collector.collect_files = MagicMock(side_effect=fake_collect)
        ds = StreamingCodeDataset(
            collector, teacher, FakeTokenizer(), Language.PYTHON, max_length=4, repos=["bad/repo", "good/repo"]
        )
        with caplog.at_level(logging.WARNING, logger="integrations.code.data"):
            items = list(ds)
        assert [i["repo"] for i in items] == ["good/repo"]
        assert "Error processing repo bad/repo: clone failed" in caplog.text

    def test_failed_critique_skips_only_that_file(self, collector):
        teacher = MagicMock(spec=CodeTeacher)
        teacher.critique.side_effect = [RuntimeError("x"), make_critique(8.0)]
        collector.collect_files = MagicMock(return_value=iter([("a.py", "aa"), ("b.py", "bb")]))
        ds = StreamingCodeDataset(
            collector, teacher, FakeTokenizer(), Language.PYTHON, max_length=4, repos=["r/r"]
        )
        assert [i["filename"] for i in ds] == ["b.py"]

    def test_critique_sample_has_language_and_filename(self, collector, teacher):
        collector.collect_files = MagicMock(return_value=iter([("a.py", "aa")]))
        ds = StreamingCodeDataset(collector, teacher, FakeTokenizer(), Language.PYTHON, max_length=4, repos=["r/r"])
        list(ds)
        sample = teacher.critique.call_args.args[0]
        assert (sample.code, sample.language, sample.filename) == ("aa", Language.PYTHON, "a.py")
