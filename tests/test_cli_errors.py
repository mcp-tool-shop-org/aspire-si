"""
Tests for the CLI's error reporting and output levels, and the error shape (aspire.errors).
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from aspire import cli
from aspire.errors import USER_ERROR, AspireError, ConfigError, missing_api_key
from aspire.teachers.base import InputValidationError

runner = CliRunner()


@pytest.fixture(autouse=True)
def _normal_level():
    cli.set_level("normal")
    yield
    cli.set_level("normal")


class TestErrorShape:
    def test_an_error_carries_code_message_hint_and_cause(self):
        cause = OSError("disk")
        error = AspireError("It broke.", code="X_BROKE", hint="Fix it.", cause=cause, retryable=True)
        assert error.to_dict() == {
            "code": "X_BROKE",
            "message": "It broke.",
            "retryable": True,
            "hint": "Fix it.",
            "cause": "OSError: disk",
        }
        assert str(error) == "It broke.\n\nFix it."
        assert error.exit_code == 2

    def test_defaults_come_from_the_class(self):
        error = ConfigError("Missing.")
        assert (error.code, error.exit_code, str(error)) == ("ASPIRE_CONFIG", USER_ERROR, "Missing.")
        assert error.to_dict() == {"code": "ASPIRE_CONFIG", "message": "Missing.", "retryable": False}
        assert AspireError("x", exit_code=7).exit_code == 7

    def test_input_validation_is_still_a_value_error(self):
        error = InputValidationError("Prompt must be a string")
        assert isinstance(error, ValueError) and isinstance(error, AspireError)
        assert error.code == "ASPIRE_INVALID_INPUT" and error.exit_code == USER_ERROR

    def test_the_missing_key_hint_names_the_variable_and_where_to_get_one(self):
        shape = missing_api_key("SOME_KEY", "SomeTeacher", "https://example.test/keys")
        assert shape["code"] == "ASPIRE_MISSING_API_KEY" and shape["exit_code"] == USER_ERROR
        assert "export SOME_KEY=" in shape["hint"] and "https://example.test/keys" in shape["hint"]

    @pytest.mark.parametrize(
        ("module", "cls", "variable"),
        [("claude", "ClaudeTeacher", "ANTHROPIC_API_KEY"), ("openai", "OpenAITeacher", "OPENAI_API_KEY")],
    )
    def test_a_teacher_without_its_key_says_how_to_set_it(self, module, cls, variable, monkeypatch):
        monkeypatch.delenv(variable, raising=False)
        teachers = __import__(f"aspire.teachers.{module}", fromlist=[cls])
        with pytest.raises(AspireError) as raised:
            getattr(teachers, cls)()
        assert raised.value.code == "ASPIRE_MISSING_API_KEY"
        assert raised.value.message == f"{variable} not found."
        assert variable in raised.value.hint


class TestReport:
    def test_an_aspire_error_prints_code_message_hint_and_cause(self, capsys):
        code = cli.report(ConfigError("Bad file.", hint="Fix the file.", cause=ValueError("line 3")))
        err = capsys.readouterr().err
        assert code == USER_ERROR
        assert "ASPIRE_CONFIG" in err and "Bad file." in err and "Fix the file." in err
        assert "Cause: ValueError: line 3" in err

    def test_an_unexpected_error_points_at_debug(self, capsys):
        assert cli.report(RuntimeError("boom")) == 2
        err = capsys.readouterr().err
        assert "ASPIRE_UNEXPECTED" in err and "RuntimeError: boom" in err and "--debug" in err
        assert "Traceback" not in err


class TestRun:
    def test_an_error_exits_with_its_code_and_no_traceback(self, capsys):
        with patch.object(cli, "app", side_effect=ConfigError("Missing.", hint="Add it.")):
            with pytest.raises(SystemExit) as exited:
                cli.run()
        assert exited.value.code == USER_ERROR
        assert "Traceback" not in capsys.readouterr().err

    def test_debug_lets_the_traceback_through(self):
        cli.set_level("debug")
        with patch.object(cli, "app", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                cli.run()

    def test_an_interrupt_exits_130(self):
        with patch.object(cli, "app", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as exited:
                cli.run()
        assert exited.value.code == 130

    def test_a_clean_run_returns(self):
        with patch.object(cli, "app", return_value=None):
            cli.run()


class TestLevels:
    def test_quiet_silences_normal_output(self):
        cli.set_level("quiet")
        assert cli.console.quiet
        cli.set_level("normal")
        assert not cli.console.quiet

    def test_quiet_reaches_the_trainer_console_once_it_is_loaded(self):
        trainer = MagicMock()
        with patch.dict(sys.modules, {"aspire.trainer": trainer}):
            cli.set_level("quiet")
            assert trainer.console.quiet is True

    def test_an_unknown_level_is_refused(self):
        with pytest.raises(ValueError, match="Unknown output level"):
            cli.set_level("loud")

    def test_verbose_prints_only_at_verbose_and_debug(self, capsys):
        cli.verbose("hidden")
        cli.set_level("verbose")
        cli.verbose("shown")
        out = capsys.readouterr().out
        assert "shown" in out and "hidden" not in out

    @pytest.mark.parametrize(
        ("flags", "level"),
        [([], "normal"), (["-q"], "quiet"), (["--verbose"], "verbose"), (["--debug"], "debug")],
    )
    def test_the_flags_set_the_level(self, flags, level):
        result = runner.invoke(cli.app, [*flags, "teachers"])
        assert result.exit_code == 0
        assert cli.state["level"] == level


class TestFiles:
    def test_a_missing_config_is_a_user_error(self, tmp_path):
        result = runner.invoke(cli.app, ["train", "--config", str(tmp_path / "nope.yaml")])
        assert isinstance(result.exception, ConfigError)
        assert "does not exist" in result.exception.message

    @pytest.mark.parametrize(
        ("content", "phrase"),
        [("{not json", "not valid JSON"), (json.dumps({"a": 1}), "not a list of strings"),
         (json.dumps(["ok", 3]), "not a list of strings")],
    )
    def test_a_bad_prompts_file_says_what_is_wrong(self, tmp_path, content, phrase):
        path = tmp_path / "prompts.json"
        path.write_text(content, encoding="utf-8")
        with patch("aspire.trainer.AspireTrainer"):
            result = runner.invoke(cli.app, ["train", "--prompts", str(path), "--output", str(tmp_path)])
        assert isinstance(result.exception, ConfigError)
        assert phrase in result.exception.message

    def test_a_missing_prompts_file_is_a_user_error(self, tmp_path):
        with patch("aspire.trainer.AspireTrainer"):
            result = runner.invoke(cli.app, ["train", "--prompts", str(tmp_path / "none.json")])
        assert isinstance(result.exception, ConfigError)
        assert "does not exist" in result.exception.message

    def test_good_prompts_reach_the_trainer_and_verbose_shows_the_settings(self, tmp_path):
        path = tmp_path / "prompts.json"
        path.write_text(json.dumps(["one", "two"]), encoding="utf-8")
        with patch("aspire.trainer.AspireTrainer") as trainer_class:
            result = runner.invoke(
                cli.app, ["--verbose", "train", "--prompts", str(path), "--output", str(tmp_path), "--geometry"]
            )
        assert result.exit_code == 0, result.output
        trainer_class.return_value.train.assert_called_once_with(["one", "two"])
        config = trainer_class.call_args.args[0]
        assert config.training.geometry_export is True
        assert '"geometry_export": true' in result.output

    def test_a_checkpoint_without_its_config_says_so(self, tmp_path):
        prompts = tmp_path / "prompts.json"
        prompts.write_text("[]", encoding="utf-8")
        result = runner.invoke(cli.app, ["evaluate", str(tmp_path), "--prompts", str(prompts)])
        assert isinstance(result.exception, ConfigError)
        assert "has no config.yaml" in result.exception.message
