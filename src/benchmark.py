from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """0 = no hits, 0.5 = partial, 1.0 = all hits."""
    if not expected:
        return 1.0
    hits = sum(1 for kw in expected if kw.lower() in answer.lower())
    if hits == 0:
        return 0.0
    return 0.5 if hits < len(expected) else 1.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score: recall weighted by response length."""
    if not answer or len(answer) < 5:
        return 0.0
    score = recall_points(answer, expected)
    if len(answer) < 20:
        score *= 0.5
    return min(1.0, score)


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: Any,
) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    memory_bytes = 0
    total_compactions = 0

    for conv in conversations:
        user_id: str = conv.get("user_id", "user")
        thread_id: str = conv.get("id", "thread")

        # Feed all turns
        for turn in conv.get("turns", []):
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        # Recall questions in a fresh thread (cross-session test)
        recall_thread = f"{thread_id}_recall"
        for rq in conv.get("recall_questions", []):
            expected: list[str] = rq.get("expected_contains", [])
            result = agent.reply(user_id, recall_thread, rq["question"])
            answer: str = result.get("response", "")
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

        # Memory size and compaction tracking
        if hasattr(agent, "memory_file_size"):
            memory_bytes = max(memory_bytes, agent.memory_file_size(user_id))
        total_compactions += agent.compaction_count(thread_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_bytes,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent Tokens",
        "Prompt Tokens",
        "Cross-session Recall",
        "Response Quality",
        "Memory Growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=headers, tablefmt="github")
    except ImportError:
        lines = [" | ".join(headers), "-" * 80]
        for row in table:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    standard_path = config.data_dir / "conversations.json"
    long_context_path = config.data_dir / "advanced_long_context.json"

    standard_convs = load_conversations(standard_path)
    long_context_convs = load_conversations(long_context_path)

    # ── Standard benchmark ────────────────────────────────────────────
    print("=" * 64)
    print("STANDARD BENCHMARK  (data/conversations.json)")
    print("=" * 64)

    baseline_std = BaselineAgent(config=config, force_offline=False)
    advanced_std = AdvancedAgent(config=config, force_offline=False)

    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, standard_convs, config),
        run_agent_benchmark("Advanced", advanced_std, standard_convs, config),
    ]
    print(format_rows(std_rows))

    # ── Long-context stress benchmark ────────────────────────────────
    print()
    print("=" * 64)
    print("LONG-CONTEXT STRESS BENCHMARK  (data/advanced_long_context.json)")
    print("=" * 64)

    baseline_lc = BaselineAgent(config=config, force_offline=False)
    advanced_lc = AdvancedAgent(config=config, force_offline=False)

    lc_rows = [
        run_agent_benchmark("Baseline", baseline_lc, long_context_convs, config),
        run_agent_benchmark("Advanced", advanced_lc, long_context_convs, config),
    ]
    print(format_rows(lc_rows))


if __name__ == "__main__":
    main()
