"""Stateless smol agent: hand-rolled tool loop against llama.cpp v1.

- non-thinking per request: chat_template_kwargs={"enable_thinking": false}
- streaming: final answer deltas are yielded live (sentence-chunked TTS
  downstream can start before the full reply is generated)
- tool calls are executed internally and the loop re-calls the model
"""
from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional

import httpx

DEFAULT_MAX_TOKENS = 300
DEFAULT_MAX_TOOL_ROUNDS = 3


class ToolRound:
    """Control signal: a round ended with tool calls (already executed)."""


class Agent:
    def __init__(
        self,
        base_url: str,
        model: str,
        persona: str,
        tools: Optional[List[dict]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        timeout: float = 120.0,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.persona = persona
        self.tools = tools
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
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
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.tools:
            payload["tools"] = self.tools
            payload["tool_choice"] = "auto"
        return payload

    def _stream_once(self, messages: List[dict]):
        """One completion. Yields (kind, value) where kind is 'text' or
        'tool_calls' (final list at end of stream)."""
        content_parts: List[str] = []
        tool_ids: Dict[int, str] = {}
        tool_names: Dict[int, str] = {}
        tool_args: Dict[int, str] = {}

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
                text = delta.get("content")
                if text:
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

    def reply(self, user_text: str, execute) -> Iterator:
        """Run the tool loop. Yields str deltas of the final answer and
        ToolRound markers after executed tool rounds. `execute(name, args)
        -> str` runs a tool."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self.persona}\nYou are a hands-free voice assistant; your "
                    "replies are spoken aloud. Keep replies to one or two short "
                    "spoken sentences. Use the available tools when they help."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        rounds = 0
        while True:
            tool_calls: Optional[List[dict]] = None
            for kind, value in self._stream_once(messages):
                if kind == "text":
                    yield value
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
            for tc in tool_calls:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = execute(tc["function"]["name"], args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    }
                )
            yield ToolRound()
