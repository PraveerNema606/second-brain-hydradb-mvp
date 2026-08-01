"""
Mode-specific system prompt overrides for the Second Brain MVP.

Each mode tweaks the assistant's instructions to suit a different
question style. All modes share the same hard rules:
  - Answer ONLY from the provided context.
  - Cite sources as [1], [2], ...
  - If the context is insufficient, return the exact fallback string.

Backward compatible: "default" reproduces the original prompt shape
(with connector-agnostic wording).
"""

# Canonical fallback when the model (or empty-context short-circuit)
# cannot answer from retrieved snippets. Connector-agnostic on purpose:
# context may come from Slack, Gmail, or structured memory.
INSUFFICIENT_CONTEXT_ANSWER = "I do not have enough context to answer that."

# Pre-Phase-2 string. Still recognized so cached / in-flight model
# replies and older clients are classified as refusals correctly.
LEGACY_INSUFFICIENT_CONTEXT_ANSWER = (
    "I do not have enough Slack context to answer that."
)

# Machine-readable retrieval / answer outcomes surfaced in debug.
OUTCOME_OK = "ok"
OUTCOME_EMPTY_CORPUS = "empty_corpus"
OUTCOME_NO_USABLE_TEXT = "no_usable_text"
OUTCOME_FILTERED_OUT = "filtered_out"
OUTCOME_LLM_REFUSAL = "llm_refusal"

# User-facing copy for retrieval failures (LLM refusal keeps the
# canonical short string so the model contract stays one exact line).
_OUTCOME_ANSWERS = {
    OUTCOME_EMPTY_CORPUS: (
        "I do not have enough context to answer that. "
        "No matching documents were found — if you recently ingested, "
        "indexing may still be in progress; try again shortly."
    ),
    OUTCOME_NO_USABLE_TEXT: (
        "I do not have enough context to answer that. "
        "Documents were returned but no usable text could be extracted."
    ),
    OUTCOME_FILTERED_OUT: (
        "I do not have enough context to answer that. "
        "Matching documents were found but none survived the applied filters."
    ),
    OUTCOME_LLM_REFUSAL: INSUFFICIENT_CONTEXT_ANSWER,
}


def is_insufficient_context_answer(text: str) -> bool:
    """True if `text` is the canonical or legacy insufficient-context reply."""
    t = (text or "").strip()
    if not t:
        return False
    if t in (INSUFFICIENT_CONTEXT_ANSWER, LEGACY_INSUFFICIENT_CONTEXT_ANSWER):
        return True
    # Outcome-specific user messages all start with the canonical prefix.
    return t.startswith(INSUFFICIENT_CONTEXT_ANSWER)


def answer_for_retrieval_outcome(outcome: str) -> str:
    """User-facing answer string for a failed retrieval outcome."""
    return _OUTCOME_ANSWERS.get(outcome, INSUFFICIENT_CONTEXT_ANSWER)


SUPPORTED_MODES = (
    "default",
    "summary",
    "decisions",
    "action_items",
    "who_said",
    "exact",
    "hybrid",
)


_HARD_RULES = f"""Rules:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the context is insufficient or unrelated, reply EXACTLY with:
  {INSUFFICIENT_CONTEXT_ANSWER}
- Cite sources inline using bracketed numbers like [1], [2] that correspond
  to the numbered snippets you used. Cite every claim that comes from the
  context.
- Do not invent people, user IDs, dates, channels, decisions, or quotes
  that are not in the context.
"""


_BASE_PROMPT = (
    "You are the Second Brain assistant. You answer questions about the "
    "user's workspace communications (Slack messages/threads and Gmail "
    "emails) using ONLY the numbered context snippets provided below the "
    "user's question.\n\n"
)


# Per-mode "goal" paragraph. Hard rules are appended to all of them.
_MODE_BODIES = {
    "default": ("Be concise. Prefer short, direct answers over restating the context."),
    "summary": (
        "Summarize the relevant context into a short briefing.\n"
        "- 3 to 6 bullet points.\n"
        "- Each bullet is one sentence, in the user's words where possible.\n"
        "- Cite the source for every bullet.\n"
        "- Do not include greetings, sign-offs, or filler."
    ),
    "decisions": (
        "Extract only DECISIONS that were made in the provided context.\n"
        "- A decision is an explicit choice, conclusion, or commitment.\n"
        "- List each decision as a single line, prefixed with '- '.\n"
        "- Cite the source for each decision.\n"
        "- If no decisions are present, reply exactly with the fallback string."
    ),
    "action_items": (
        "Extract ACTION ITEMS only.\n"
        "- An action item is a concrete task someone is doing, will do, or "
        "has been asked to do.\n"
        "- Format each line as: '- [owner if known] task description [N]'.\n"
        "- Cite the source for each item.\n"
        "- If no action items are present, reply exactly with the fallback string."
    ),
    "who_said": (
        "Identify WHO SAID WHAT relevant to the question.\n"
        "- Format each quote as: '- <user name>: \"<verbatim quote>\" [N]'.\n"
        "- Use only quotes that are present in the context — do not paraphrase.\n"
        "- Cite the source for each quote.\n"
        "- If no relevant quotes exist, reply exactly with the fallback string."
    ),
    "exact": (
        "You are operating in EXACT-MATCH mode. The context snippets shown "
        "below were selected because they contain literal keyword matches "
        "from the user's question.\n"
        "- Quote or directly reference the matching phrases from the context.\n"
        "- Stay close to the wording of the source. Do not paraphrase "
        "loosely or add nuance not present in the snippets.\n"
        "- Cite every claim. Prefer terse answers over long ones."
    ),
    "hybrid": (
        "You are operating in HYBRID-RETRIEVAL mode. The context combines "
        "semantically-similar snippets with snippets that contain literal "
        "keyword matches from the question.\n"
        "- Prefer evidence that includes the user's exact terms when "
        "available, but you may still use semantically related context.\n"
        "- Cite every claim. Prefer concise, direct answers."
    ),
}


def system_prompt_for_mode(mode: str) -> str:
    """
    Return the system prompt for a mode. Unknown modes fall back to
    'default' silently so a client typo doesn't fail the request — the
    Pydantic Literal in main.py is what actually enforces the allowed set.
    """
    body = _MODE_BODIES.get(mode, _MODE_BODIES["default"])
    return f"{_BASE_PROMPT}{body}\n\n{_HARD_RULES}"


# ---------------------------------------------------------------------- #
# Conversation history formatting
# ---------------------------------------------------------------------- #
# Per-turn cap so a long prior assistant answer doesn't blow up the prompt.
# Approximate; we slice on characters, not tokens, but that's fine for an
# MVP — the LLM still has plenty of headroom even after a few truncated
# turns plus the retrieved context.
_HISTORY_TURN_CHAR_CAP = 800


def format_conversation_history(history) -> str:
    """
    Build a short "Recent conversation context" section to prepend to the
    user-turn message in the LLM call.

    `history` is expected to be a list of dicts (or Pydantic models with
    `.role` / `.content`) representing recent USER and ASSISTANT turns,
    in chronological order (oldest first). The current question is NOT
    in this list — main.py supplies that separately.

    Returns an empty string when there's nothing to include, so callers
    can do `f"{format_conversation_history(h)}Question: ..."` without
    worrying about an extra blank line in the common case.
    """
    if not history:
        return ""

    lines = []
    for msg in history:
        # Accept either dicts or pydantic models with attribute access.
        if isinstance(msg, dict):
            role = (msg.get("role") or "").strip().lower()
            content = (msg.get("content") or "").strip()
        else:
            role = (getattr(msg, "role", "") or "").strip().lower()
            content = (getattr(msg, "content", "") or "").strip()

        if role not in ("user", "assistant") or not content:
            continue

        # Truncate per-turn so very long assistant replies don't dominate.
        if len(content) > _HISTORY_TURN_CHAR_CAP:
            content = content[:_HISTORY_TURN_CHAR_CAP].rstrip() + " […]"

        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")

    if not lines:
        return ""

    # The preamble is explicit about purpose: history is for resolving
    # references ONLY, not new evidence. Hard rules (cite from context,
    # fallback string, etc.) still apply.
    return (
        "Recent conversation context (use ONLY to resolve references "
        "like 'he', 'that', 'the earlier discussion' — do NOT cite from "
        "this section or treat it as new evidence):\n" + "\n".join(lines) + "\n\n"
    )
