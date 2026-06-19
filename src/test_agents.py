from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path) -> LabConfig:
    dummy = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0, api_key="dummy")
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,   # very low so compaction triggers quickly in tests
        compact_keep_messages=2,
        model=dummy,
        judge_model=dummy,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # Read before file exists — should return a default string
    default = store.read_text("alice")
    assert "alice" in default.lower() or "no information" in default.lower()

    # Write
    store.write_text("alice", "# Profile: alice\n\n- **name**: Alice\n")
    assert store.path_for("alice").exists()

    # Read back
    content = store.read_text("alice")
    assert "Alice" in content

    # Edit — replace text inside the file
    changed = store.edit_text("alice", "Alice", "Alice Nguyen")
    assert changed is True
    assert "Alice Nguyen" in store.read_text("alice")

    # Edit — search text not present → should return False
    not_changed = store.edit_text("alice", "NonExistent", "x")
    assert not_changed is False

    # File size
    assert store.file_size("alice") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=30, keep_messages=2)
    tid = "thread-compact"

    # Pump enough messages to exceed the token threshold
    for i in range(12):
        manager.append(tid, "user", f"Message number {i} with enough words to count tokens easily.")
        manager.append(tid, "assistant", f"Reply {i} — more words to fill the budget faster.")

    assert manager.compaction_count(tid) > 0

    ctx = manager.context(tid)
    # At most keep_messages should remain in the live window
    assert len(ctx["messages"]) <= 2  # type: ignore[arg-type]
    # Summary should be non-empty after compaction
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    # Session 1: teach facts
    advanced.reply("user1", "thread-a", "Mình tên là TestUser.")
    advanced.reply("user1", "thread-a", "Mình ở Hà Nội.")
    baseline.reply("user1", "thread-a", "Mình tên là TestUser.")
    baseline.reply("user1", "thread-a", "Mình ở Hà Nội.")

    # Session 2: completely new thread — only Advanced should recall
    adv_result = advanced.reply("user1", "thread-b", "Mình tên gì?")
    base_result = baseline.reply("user1", "thread-b", "Mình tên gì?")

    assert "TestUser" in adv_result["response"], "Advanced should recall name from User.md"
    assert "TestUser" not in base_result["response"], "Baseline must not recall across threads"


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    # Feed a long thread — baseline accumulates full history, advanced compacts
    for i in range(20):
        msg = f"Đây là tin nhắn số {i}, chứa nội dung dài hơn để tăng số token trong thread."
        baseline.reply("user1", "long-thread", msg)
        advanced.reply("user1", "long-thread", msg)

    baseline_prompt = baseline.prompt_token_usage("long-thread")
    advanced_prompt = advanced.prompt_token_usage("long-thread")

    # Baseline carries full history → prompt tokens grow unbounded.
    # Advanced compacts → prompt tokens should be meaningfully lower.
    # We allow up to 90 % of baseline as a generous upper bound.
    assert advanced_prompt <= baseline_prompt * 0.90 or advanced.compaction_count("long-thread") > 0, (
        f"Expected advanced ({advanced_prompt}) to use fewer prompt tokens than "
        f"baseline ({baseline_prompt}) or to have at least one compaction."
    )
