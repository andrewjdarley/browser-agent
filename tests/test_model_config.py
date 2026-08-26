"""Tests for model_config.py - the central, env-var-overridable place for
which model each part of the app uses (previously scattered: find's model
was hardcoded inline, subagent/verify silently inherited the coordinator's
model with no way to change that)."""

import importlib

import pytest

from browser_use_demo import model_config


@pytest.fixture(autouse=True)
def _reset_model_config_after_test():
    """Some tests reload the module against a monkeypatched env var: put it
    back in sync with the real environment afterward so state doesn't leak
    between tests."""
    yield
    importlib.reload(model_config)


class TestResolve:
    """resolve() is the shared override-or-fall-back-to-coordinator logic
    used for both SUBAGENT_MODEL and VERIFY_MODEL in loop.py."""

    def test_override_wins_when_set(self):
        assert model_config.resolve("claude-haiku-4-5-20251001", "claude-opus-5") == "claude-haiku-4-5-20251001"

    def test_falls_back_to_coordinator_model_when_none(self):
        assert model_config.resolve(None, "claude-opus-5") == "claude-opus-5"


class TestDefaults:
    def test_find_model_defaults_to_haiku(self, monkeypatch):
        monkeypatch.delenv("BROWSER_USE_FIND_MODEL", raising=False)
        reloaded = importlib.reload(model_config)
        assert reloaded.FIND_MODEL == "claude-haiku-4-5-20251001"

    def test_subagent_and_verify_model_default_to_none(self, monkeypatch):
        monkeypatch.delenv("BROWSER_USE_SUBAGENT_MODEL", raising=False)
        monkeypatch.delenv("BROWSER_USE_VERIFY_MODEL", raising=False)
        reloaded = importlib.reload(model_config)
        assert reloaded.SUBAGENT_MODEL is None
        assert reloaded.VERIFY_MODEL is None


class TestEnvOverrides:
    def test_find_model_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("BROWSER_USE_FIND_MODEL", "claude-opus-5")
        reloaded = importlib.reload(model_config)
        assert reloaded.FIND_MODEL == "claude-opus-5"

    def test_subagent_model_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("BROWSER_USE_SUBAGENT_MODEL", "claude-haiku-4-5-20251001")
        reloaded = importlib.reload(model_config)
        assert reloaded.SUBAGENT_MODEL == "claude-haiku-4-5-20251001"

    def test_verify_model_overridable_via_env(self, monkeypatch):
        monkeypatch.setenv("BROWSER_USE_VERIFY_MODEL", "claude-opus-5")
        reloaded = importlib.reload(model_config)
        assert reloaded.VERIFY_MODEL == "claude-opus-5"

    def test_empty_string_env_value_treated_as_unset(self, monkeypatch):
        """An empty override should still fall back to the coordinator's
        model, not resolve to an empty model string."""
        monkeypatch.setenv("BROWSER_USE_SUBAGENT_MODEL", "")
        reloaded = importlib.reload(model_config)
        assert reloaded.SUBAGENT_MODEL is None
