from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B — three-layer memory: short-term + User.md + compact memory.

    Short-term  : messages kept inside CompactMemoryManager per thread.
    Persistent  : User.md per user, updated whenever stable facts are detected.
    Compact     : older messages are summarised when token budget is exceeded.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Persist stable facts into User.md
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2. Append user message to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context carried into this turn
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        # 4. Generate a response that uses persisted memory
        response = self._offline_response(user_id, thread_id, message)

        # 5. Persist assistant reply and update counters
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary_text = str(ctx.get("summary", ""))
        messages_text = " ".join(
            m["content"] for m in ctx.get("messages", [])  # type: ignore[union-attr]
        )
        return estimate_tokens(profile_text + summary_text + messages_text)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        msg_lower = message.lower()

        def _has(*keywords: str) -> bool:
            return all(k in msg_lower for k in keywords)

        # Targeted recall questions
        if _has("tên") and "?" in message:
            if facts.get("name"):
                return f"[Advanced] Tên bạn là {facts['name']}."

        if _has("ở đâu") or _has("nơi ở"):
            if facts.get("location"):
                return f"[Advanced] Bạn đang ở {facts['location']}."

        if _has("nghề") or _has("làm gì"):
            if facts.get("profession"):
                return f"[Advanced] Bạn đang làm {facts['profession']}."

        if _has("đồ uống") or _has("uống gì"):
            if facts.get("favorite_drink"):
                return f"[Advanced] Đồ uống yêu thích của bạn là {facts['favorite_drink']}."

        if _has("món ăn") or _has("ăn gì"):
            if facts.get("favorite_food"):
                return f"[Advanced] Món ăn yêu thích của bạn là {facts['favorite_food']}."

        if _has("style") or (_has("trả lời") and _has("thích")):
            if facts.get("response_style"):
                return f"[Advanced] Style bạn thích: {facts['response_style']}."

        if _has("nuôi") or _has("corgi") or _has("bơ"):
            if facts.get("pet"):
                return f"[Advanced] Bạn nuôi {facts['pet']}."

        # Composite recall / summary
        if _has("nhắc lại") or _has("tóm tắt") or _has("mô tả"):
            parts = []
            label_map = {
                "name": "Tên",
                "location": "Nơi ở",
                "profession": "Nghề nghiệp",
                "favorite_drink": "Đồ uống",
                "favorite_food": "Món ăn",
                "response_style": "Style",
                "pet": "Thú cưng",
            }
            for key, label in label_map.items():
                if facts.get(key):
                    parts.append(f"{label}: {facts[key]}")
            if parts:
                return "[Advanced] " + " | ".join(parts)

        # Who-am-I / summary question
        if ("ai" in msg_lower and "?" in message) or _has("biết") and _has("không"):
            parts = []
            for key in ("name", "profession", "location", "favorite_drink", "response_style"):
                if facts.get(key):
                    parts.append(facts[key])
            if parts:
                return "[Advanced] " + ", ".join(parts) + "."

        return (
            f"[Advanced] Nhận được: {message[:80]}{'...' if len(message) > 80 else ''}"
        )

    # ------------------------------------------------------------------
    # Live (LangChain) path
    # ------------------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        profile_text = self.profile_store.read_text(user_id)
        system_content = (
            "Bạn là assistant nhớ thông tin người dùng qua nhiều phiên.\n"
            f"Đây là profile hiện tại của người dùng:\n\n{profile_text}"
        )
        config_dict = {"configurable": {"thread_id": thread_id}}
        result = self.langchain_agent.invoke(
            {"messages": [SystemMessage(content=system_content), HumanMessage(content=message)]},
            config=config_dict,
        )
        response: str = result["messages"][-1].content

        self.compact_memory.append(thread_id, "user", message)
        self.compact_memory.append(thread_id, "assistant", response)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        agent_tokens = estimate_tokens(response)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self) -> None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)
            self.langchain_agent = create_react_agent(llm, tools=[], checkpointer=MemorySaver())
        except Exception:
            self.langchain_agent = None
