"""Offline tests for the no-ChatGPT pool builder."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fugu_swarm.pool import is_openai, slot_csv, required_env


def test_is_openai_detects_gpt_family():
    assert is_openai("openai/gpt-4o-mini")
    assert is_openai("gpt-5")
    assert is_openai("azure/gpt-4")
    assert is_openai("some-proxy/chatgpt-latest")
    assert is_openai("openai/o3-mini")
    assert is_openai("openai/codex-mini")


def test_is_openai_allows_others():
    for ok in ["anthropic/claude-opus-4-8", "gemini/gemini-3.1-pro",
               "deepseek/deepseek-v4", "dashscope/qwen3-32b",
               "openrouter/z-ai/glm-5", "novita/moonshotai/kimi-k2"]:
        assert not is_openai(ok), ok


def test_slot_csv_blocks_openai_by_default():
    try:
        slot_csv(["anthropic/claude-opus-4-8", "openai/gpt-4o"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "openai/gpt-4o" in str(e)


def test_slot_csv_allows_override():
    csv = slot_csv(["anthropic/claude-opus-4-8", "openai/gpt-4o"], allow_openai=True)
    assert csv.count(",") == 1


def test_slot_csv_happy_path_and_empty():
    csv = slot_csv(["anthropic/claude-opus-4-8", "deepseek/deepseek-v4"])
    assert csv == "anthropic/claude-opus-4-8,deepseek/deepseek-v4"
    try:
        slot_csv([])
        assert False
    except ValueError:
        pass


def test_required_env_maps_providers():
    env = required_env(["anthropic/claude-opus-4-8", "deepseek/deepseek-v4"])
    assert "ANTHROPIC_API_KEY" in env and "DEEPSEEK_API_KEY" in env


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all pool tests passed")
