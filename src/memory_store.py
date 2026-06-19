from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class UserProfileStore:
    """Persistent per-user profile backed by a markdown file (User.md)."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^\w\-]", "_", user_id)
        return self.root_dir / f"{safe_id}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return f"# Profile: {user_id}\n\n(no information yet)\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        self.write_text(user_id, current.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse `- **key**: value` lines from the profile."""
        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            if line.startswith("- **") and "**:" in line:
                key_part, _, value_part = line.partition("**:")
                key = key_part.lstrip("- **").strip()
                result[key] = value_part.strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        current = self.read_text(user_id)
        marker = f"- **{key}**:"
        new_line = f"- **{key}**: {value}"

        if marker in current:
            lines = [new_line if ln.startswith(marker) else ln for ln in current.splitlines()]
            self.write_text(user_id, "\n".join(lines) + "\n")
        else:
            if "(no information yet)" in current:
                content = f"# Profile: {user_id}\n\n{new_line}\n"
            else:
                content = current.rstrip() + f"\n{new_line}\n"
            self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from a user message using regex heuristics."""

    # Skip pure question turns
    stripped = message.strip()
    if stripped.endswith("?") and len(stripped.split()) < 15:
        return {}

    facts: dict[str, str] = {}

    # Name
    for pattern in [
        r"mình tên(?:\s+là)?\s+([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9]+)",
        r"tên(?:\s+mình)?\s+là\s+([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9]+)",
    ]:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            facts["name"] = m.group(1).strip()
            break

    # Location — look for explicit "now at X" phrasing first
    for pattern in [
        r"(?:giờ|đang|hiện tại)\s+(?:mình\s+)?(?:đang\s+)?ở\s+([A-ZÀ-Ỵ][a-zà-ỵ]+(?:\s[A-ZÀ-Ỵ][a-zà-ỵ]+)?)",
        r"mình\s+(?:đang\s+)?ở\s+([A-ZÀ-Ỵ][a-zà-ỵ]+(?:\s[A-ZÀ-Ỵ][a-zà-ỵ]+)?)",
        r"\bở\s+([A-ZÀ-Ỵ][a-zà-ỵ]+(?:\s[A-ZÀ-Ỵ][a-zà-ỵ]+)?)\b",
    ]:
        m = re.search(pattern, message)
        if m:
            loc = m.group(1).strip()
            if loc not in {"lại", "đây", "kia", "đó"} and len(loc.split()) <= 3:
                facts["location"] = loc
                break

    # Profession
    for pattern in [
        r"chuyển\s+sang\s+([A-Za-zÀ-ỹ\s]+?(?:engineer|developer|analyst|designer|researcher))",
        r"làm\s+([A-Za-zÀ-ỹ\s]+?(?:engineer|developer|analyst|designer|researcher))",
    ]:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            facts["profession"] = m.group(1).strip()
            break

    # Favorite drink
    for pattern in [
        r"đồ uống yêu thích(?:\s+là)?\s+([a-zA-ZÀ-ỹ\s]+?)(?:\.|,|$)",
        r"thích(?:\s+uống)?\s+([a-zA-ZÀ-ỹ\s]*?(?:sữa đá|đen nóng|đen đá|nóng|kem))",
    ]:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            facts["favorite_drink"] = m.group(1).strip().rstrip(".,")
            break

    # Favorite food
    for pattern in [
        r"món ăn yêu thích(?:\s+là)?\s+([a-zA-ZÀ-ỹ\s]+?)(?:\.|,|$)",
        r"thích ăn\s+([a-zA-ZÀ-ỹ\s]+?)(?:\.|,|$)",
    ]:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            facts["favorite_food"] = m.group(1).strip()
            break

    # Response style
    if re.search(r"(?:muốn|thích)\s+(?:bạn\s+)?(?:trả lời|câu trả lời)\s+ngắn", message, re.IGNORECASE):
        facts["response_style"] = "ngắn gọn, rõ ý, có ví dụ thực tế"

    # Pet
    m = re.search(r"nuôi\s+(?:một\s+)?(?:bé\s+)?(\w+)\s+tên\s+(\w+)", message, re.IGNORECASE)
    if m:
        facts["pet"] = f"{m.group(1)} tên {m.group(2)}"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    if not messages:
        return ""
    items = messages[-max_items:]
    parts = [f"[{m.get('role', '')}]: {m.get('content', '')[:200]}" for m in items]
    return "Tóm tắt hội thoại trước:\n" + "\n".join(parts)


@dataclass
class CompactMemoryManager:
    """Keep recent messages in full; summarise older ones when token budget is exceeded."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._init_thread(thread_id)
        thread = self.state[thread_id]
        thread["messages"].append({"role": role, "content": content})  # type: ignore[index]

        total_text = (
            " ".join(m["content"] for m in thread["messages"])  # type: ignore[index]
            + str(thread["summary"])
        )
        if estimate_tokens(total_text) > self.threshold_tokens:
            self._compact(thread_id)

    def _compact(self, thread_id: str) -> None:
        thread = self.state[thread_id]
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]

        if len(messages) <= self.keep_messages:
            return

        old_messages = messages[: -self.keep_messages]
        recent_messages = messages[-self.keep_messages :]

        old_summary = str(thread["summary"])
        parts = [old_summary] if old_summary else []
        parts.append(summarize_messages(old_messages))

        thread["summary"] = "\n".join(parts)
        thread["messages"] = recent_messages
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        self._init_thread(thread_id)
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        self._init_thread(thread_id)
        return int(self.state[thread_id]["compactions"])  # type: ignore[arg-type]
