"""Smol agent: hand-rolled tool loop against llama.cpp v1.

- per-request thinking toggle: chat_template_kwargs={"enable_thinking": ...};
  with thinking on, reasoning tokens stream before the spoken answer and are
  surfaced as ReasoningDelta (never spoken, never stored)
- streaming: final answer deltas are yielded live (sentence-chunked TTS
  downstream can start before the full reply is generated)
- tool calls are executed internally (in parallel within a round) and the
  loop re-calls the model; tool exceptions become model-visible error text
- `reply()` accepts prior conversation history (see app/conversation.py)
- `summarize()` is a non-streaming helper for conversation compaction
- `dream_pass()` is a non-streaming LLM memory-consolidation pass (dreaming)
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional

import httpx

DEFAULT_MAX_TOKENS = 300
DEFAULT_MAX_TOOL_ROUNDS = 8
SUMMARY_MAX_TOKENS = 200
DREAM_MAX_TOKENS = 600
DEFAULT_TOOL_RETRIES = 2

DREAM_PROMPT = (
    "You are vivo's memory consolidator. You are given the memory archive and "
    "the facts it considers most worth keeping. Return a JSON object with "
    "exactly these keys:\n"
    '- "summary": one short paragraph (2-4 sentences) of what matters most '
    "about this person and their projects right now.\n"
    '- "new_facts": a list of 0-5 short factual sentences (each under 200 '
    "characters) implied by the archive but not yet stored as facts. Only "
    "genuinely new, durable facts; no duplicates of existing facts, no "
    "transient reminders or task output.\n"
    '- "connections": a list of 0-5 short sentences linking facts to each '
    "other (shared projects, people, or places).\n"
    '- "pruned": a list of fact ids that are stale, contradicted, or low '
    "value and can be deleted. Never list a fact marked (core).\n"
    "Reply with the JSON object only: no markdown fences, no commentary."
)


class ToolRound:
    """Control signal: a round ended with tool calls (already executed)."""


class ReasoningDelta:
    """Control signal: a chunk of model reasoning (thinking) streamed before
    the spoken answer. Never spoken, never stored in history."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"ReasoningDelta({self.text!r})"


class Agent:
    def __init__(
        self,
        base_url: str,
        model: str,
        persona: str,
        tools: Optional[List[dict]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        tool_retries: int = DEFAULT_TOOL_RETRIES,
        thinking: bool = False,
        system_prompt: str = "",
        user_profile: str = "",
        timeout: float = 120.0,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.persona = persona
        self.system_prompt = system_prompt
        self.user_profile = user_profile
        self.tools = tools
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.tool_retries = max(int(tool_retries), 0)
        self.client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _payload(self, messages: List[dict]) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if self.tools:
            payload["tools"] = self.tools
            payload["tool_choice"] = "auto"
        return payload

    def _stream_once(self, messages: List[dict]):
        """One completion. Yields (kind, value) where kind is 'reasoning'
        (thinking deltas, when enabled), 'text', or 'tool_calls' (final list
        at end of stream)."""
        for attempt in range(self.tool_retries + 1):
            content_parts: List[str] = []
            tool_ids: Dict[int, str] = {}
            tool_names: Dict[int, str] = {}
            tool_args: Dict[int, str] = {}
            emitted = False
            try:
                with self.client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=self._payload(messages),
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            ev = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = ev.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        # Reasoning streams in `reasoning` (llama.cpp) or
                        # `reasoning_content` (OpenAI-style servers); never spoken.
                        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                        if reasoning:
                            emitted = True
                            yield "reasoning", reasoning
                        text = delta.get("content")
                        if text:
                            emitted = True
                            content_parts.append(text)
                            yield "text", text
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            if tc.get("id"):
                                tool_ids[idx] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                tool_names[idx] = fn["name"]
                            if fn.get("arguments"):
                                tool_args[idx] = tool_args.get(idx, "") + fn["arguments"]
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status >= 500 and not emitted and attempt < self.tool_retries:
                    continue
                raise

            tool_calls = []
            for idx in sorted(tool_names):
                tool_calls.append(
                    {
                        "id": tool_ids.get(idx) or f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": tool_names[idx],
                            "arguments": tool_args.get(idx, "{}"),
                        },
                    }
                )
            yield "tool_calls", tool_calls
            return

    def reply(
        self, user_text: str, execute, history: Optional[List[dict]] = None
    ) -> Iterator:
        """Run the tool loop. Yields str deltas of the final answer and
        ToolRound markers after executed tool rounds. `execute(name, args)
        -> str` runs a tool. `history` is prior OpenAI-style messages
        (system/user/assistant), e.g. from Conversation.messages()."""
        system = self.persona
        if self.system_prompt:
            system = f"{self.persona}\n{self.system_prompt}"
        if self.user_profile:
            system = f"{system}\n{self.user_profile}"
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        rounds = 0
        while True:
            tool_calls: Optional[List[dict]] = None
            for kind, value in self._stream_once(messages):
                if kind == "text":
                    yield value
                elif kind == "reasoning":
                    yield ReasoningDelta(value)
                else:
                    tool_calls = value
            if not tool_calls or rounds >= self.max_tool_rounds:
                return
            rounds += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                }
            )

            def run_tool(tc: dict) -> str:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    return str(execute(tc["function"]["name"], args))
                except Exception as e:  # noqa: BLE001 - model-visible error text
                    return f"error: {type(e).__name__}: {e}"

            # parallel within a round; results stay in tool_call order
            with ThreadPoolExecutor(max_workers=min(4, len(tool_calls))) as pool:
                results = list(pool.map(run_tool, tool_calls))
            for tc, result in zip(tool_calls, results):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
            yield ToolRound()

    def summarize(self, messages: List[dict]) -> str:
        """Non-streaming summary of a conversation slice (compaction)."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a memory consolidator. Summarise the "
                        "conversation in a few short factual sentences: what "
                        "was said, decided, or requested; keep names, numbers, "
                        "dates, and preferences. Do not answer questions that "
                        "appear in it. Reply with the summary only."
                    ),
                },
                *messages,
            ],
            "stream": False,
            "max_tokens": SUMMARY_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        r.raise_for_status()
        return (r.json()["choices"][0]["message"].get("content") or "").strip()

    def dream_pass(self, archive_text: str, candidates: List[dict]) -> dict:
        """Non-streaming LLM dream over the memory archive.

        Returns {"summary", "new_facts", "connections", "pruned"}. Unparseable
        output degrades to empty lists (and the raw text as summary) instead
        of raising, so a chatty model can never kill a dream pass."""
        cand_lines = "\n".join(
            f'- {item["text"]} (id: {item["id"]})'
            + (" (core)" if item.get("core") else "")
            for item in candidates
        ) or "(no saved facts yet)"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DREAM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Memory archive:\n{archive_text}\n\n"
                        f"Candidates to keep or prune:\n{cand_lines}"
                    ),
                },
            ],
            "stream": False,
            "max_tokens": DREAM_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        r.raise_for_status()
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        return parse_dream_response(content)


def parse_dream_response(content: str) -> dict:
    """Extract the dream JSON object from model output, tolerating fences and
    surrounding commentary. Missing or broken output degrades gracefully."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    raw = None
    if start != -1 and end > start:
        try:
            raw = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raw = None
    if not isinstance(raw, dict):
        return {
            "summary": text[:500],
            "new_facts": [],
            "connections": [],
            "pruned": [],
        }

    def as_list(value) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    return {
        "summary": str(raw.get("summary") or "").strip(),
        "new_facts": as_list(raw.get("new_facts"))[:10],
        "connections": as_list(raw.get("connections"))[:10],
        "pruned": as_list(raw.get("pruned"))[:20],
    }
