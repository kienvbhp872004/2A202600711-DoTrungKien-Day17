# Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc memory system](#kiến-trúc-memory-system)
- [Cấu trúc codebase](#cấu-trúc-codebase)
- [Setup môi trường](#setup-môi-trường)
- [Kết quả benchmark](#kết-quả-benchmark)
  - [Standard Benchmark](#standard-benchmark)
  - [Long-Context Stress Benchmark](#long-context-stress-benchmark)
  - [Phân tích kết quả](#phân-tích-kết-quả)
- [Bonus: Conflict Handling](#bonus-conflict-handling)
- [Provider hỗ trợ](#provider-hỗ-trợ)
- [Tài liệu](#tài-liệu)

---

## Tổng quan

Bài lab này tập trung vào câu hỏi thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** trong khi vẫn kiểm soát được chi phí token.

Hai agent được xây dựng và so sánh:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu không phải chỉ là "agent nhớ nhiều hơn", mà là hiểu rõ trade-off giữa độ nhớ dài hạn, chất lượng phản hồi, chi phí token, và độ phức tạp của hệ thống memory.

---

## Kiến trúc memory system

<!-- TODO: thêm sơ đồ kiến trúc 3 lớp memory (short-term → compact → User.md) -->
<!-- Gợi ý: diagram dạng flow từ trái sang phải, mỗi lớp một màu khác nhau -->

| Lớp | Tên | Lưu ở đâu | Tồn tại qua session mới? |
|---|---|---|---|
| 1 | Short-term memory | RAM (LangGraph state) | Không |
| 2 | Persistent memory | `User.md` trên disk | Có |
| 3 | Compact memory | RAM, tự nén khi quá ngưỡng | Không (nhưng giảm token cost) |

<!-- TODO: thêm ảnh chụp màn hình file User.md sau khi benchmark chạy xong -->
<!-- Gợi ý: so sánh User.md trước và sau khi có correction (Đà Nẵng → Huế) -->

---

## Cấu trúc codebase

```
.
├── src/
│   ├── config.py            # LabConfig + load_config()
│   ├── model_provider.py    # build_chat_model() cho 6 provider
│   ├── memory_store.py      # UserProfileStore + CompactMemoryManager
│   ├── agent_baseline.py    # Agent A — session memory only
│   ├── agent_advanced.py    # Agent B — 3-layer memory
│   ├── benchmark.py         # Standard + Long-context benchmark
│   └── test_agents.py       # 4 behavioral tests
├── data/
│   ├── conversations.json         # 10 hội thoại tiếng Việt
│   └── advanced_long_context.json # stress test dataset
├── state/
│   └── profiles/            # User.md files (sinh ra khi chạy)
└── .env                     # API keys (không commit)
```

---

## Setup môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install langchain langgraph langchain-openai langchain-google-genai \
    langchain-anthropic langchain-ollama langchain-openrouter \
    python-dotenv tabulate pytest
```

Tạo file `.env` ở root:

```
LLM_PROVIDER=custom
LLM_MODEL=your-model-name
CUSTOM_BASE_URL=https://your-gateway/v1
CUSTOM_API_KEY=your-api-key
```

Chạy test:

```bash
python -m pytest src/test_agents.py -v
```

Chạy benchmark:

```bash
python src/benchmark.py
```

---

## Kết quả benchmark

<!-- TODO: thêm ảnh chụp màn hình terminal khi benchmark chạy xong -->
<!-- Gợi ý: 2 bảng cạnh nhau — standard và long-context -->

### Standard Benchmark

Dataset: `data/conversations.json` — 10 hội thoại, mỗi hội thoại 10 lượt, có correction và follow-up.

| Agent | Agent Tokens | Prompt Tokens | Cross-session Recall | Response Quality | Memory Growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 41,843 | 200,897 | 0.32 | 0.32 | 0 | 0 |
| Advanced | 21,743 | 76,755 | **0.75** | **0.75** | 306 | 76 |

### Long-Context Stress Benchmark

Dataset: `data/advanced_long_context.json` — hội thoại rất dài để làm lộ tác động của compact memory.

| Agent | Agent Tokens | Prompt Tokens | Cross-session Recall | Response Quality | Memory Growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 5,351 | 57,374 | 0.00 | 0.00 | 0 | 0 |
| Advanced | 6,221 | **29,669** | **0.50** | **0.50** | 178 | 28 |

---

## Phân tích kết quả

### 1. Tại sao Advanced có recall cao hơn Baseline?

Baseline chỉ lưu lịch sử trong RAM theo `thread_id`. Khi recall question được hỏi ở thread mới (`conv-01_recall`), `sessions["conv-01_recall"]` hoàn toàn trống — agent không biết gì về những gì đã nói ở `conv-01`.

Advanced lưu facts ổn định vào `User.md` ngay khi nhận được từ user. Thread mới vẫn đọc được file này. Kết quả: recall 0.75 vs 0.32 ở standard benchmark, và 0.50 vs 0.00 ở stress test.

### 2. Tại sao Advanced dùng ít prompt tokens hơn mặc dù có thêm User.md?

Ở **standard benchmark**, Advanced dùng ít hơn 2.6× prompt tokens (76k vs 200k). Lý do: compact memory kích hoạt 76 lần, liên tục tóm tắt lịch sử cũ và chỉ giữ lại vài message gần nhất. Baseline không compact nên phải carry toàn bộ lịch sử ngày càng dài qua mỗi lượt.

Ở **long-context stress**, Advanced dùng ít hơn 1.9× (29k vs 57k) và compact 28 lần. Tác động rõ ràng hơn khi thread càng dài.

### 3. Tại sao ở hội thoại ngắn Advanced có thể tốn hơn?

Mỗi lượt Advanced phải carry thêm nội dung `User.md` vào context. Với hội thoại ngắn (ít turns), overhead này chưa được bù đắp bởi compact. Đây là trade-off có chủ ý: trả thêm chi phí nhỏ ở đầu để được recall tốt hơn về sau.

### 4. Memory file tăng trưởng và rủi ro

`User.md` tăng theo số facts được ghi. Với 10 hội thoại, file đạt 306 bytes — con số nhỏ. Nhưng theo thời gian nếu không có cơ chế dọn dẹp, file có thể phình ra và tốn context mỗi lượt. Rủi ro thực tế:

- Lưu sai fact khi user đặt câu hỏi thay vì cung cấp thông tin
- Fact cũ không bị xoá khi user không nói rõ là đính chính
- File quá dài làm tăng prompt tokens, triệt tiêu lợi ích của compact

---

## Bonus: Conflict Handling

Khi user đính chính thông tin cũ (`conv-03`: đổi từ Đà Nẵng sang Huế, `conv-06`: đổi từ backend engineer sang MLOps engineer), `upsert_fact()` cập nhật đúng chỗ thay vì thêm dòng mới:

```python
# User.md trước correction
- **location**: Đà Nẵng

# Sau khi upsert_fact("location", "Huế")
- **location**: Huế   ← ghi đè, không duplicate
```

Điều này giải quyết vấn đề **fact conflict** — agent không giữ đồng thời hai giá trị mâu thuẫn cho cùng một key. Kết quả: recall question "Hiện tại mình đang ở đâu?" trả về đúng "Huế" thay vì "Đà Nẵng".

**Hạn chế hiện tại:** nếu user nói *"Mình từng ở Đà Nẵng"* (câu hỏi/ví dụ, không phải fact mới), regex vẫn có thể nhận nhầm là location update. Để fix cần thêm confidence threshold — chỉ upsert khi pattern đủ chắc chắn là fact mới.

<!-- TODO: thêm ảnh so sánh User.md trước và sau correction để minh hoạ conflict handling -->

---

## Provider hỗ trợ

| Provider | Package | Env var |
|---|---|---|
| `openai` | `langchain-openai` | `OPENAI_API_KEY` |
| `custom` | `langchain-openai` | `CUSTOM_BASE_URL`, `CUSTOM_API_KEY` |
| `gemini` | `langchain-google-genai` | `GEMINI_API_KEY` |
| `anthropic` | `langchain-anthropic` | `ANTHROPIC_API_KEY` |
| `ollama` | `langchain-ollama` | `OLLAMA_BASE_URL` |
| `openrouter` | `langchain-openrouter` | `OPENROUTER_API_KEY` |

---

## Tài liệu

- [Guide.md](Guide.md): hướng dẫn từng bước hoàn thành lab
- [Rubric.md](Rubric.md): tiêu chí chấm điểm và bonus
