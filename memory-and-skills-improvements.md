# Skills

Skills are reusable, task-specific instruction packages. They give the agent a
reliable recipe for work that is too detailed, brittle, or recurrent to encode
in the base prompt: for example, a Gloucester weather briefing, a daily
summary, or an established project workflow.

## Storage and discovery

- Store each skill at `data/skills/<skill-name>/SKILL.md`. Use lowercase,
   hyphenated directory names such as `gloucester-weather`.
- Discover skills at startup and inject only a compact index into the system
   prompt: name and one-line description. Keep the entire index below roughly
   500 tokens.
- Load the full `SKILL.md` only when the request clearly matches its
   description or the user names the skill. The agent should not preload every
   skill into every turn.
- Expose a `list_skills` tool for discovery and a `read_skill` tool for loading
   the selected instructions. Both should be read-only and restricted to the
   skills directory.

## Skill format

Every skill uses a short YAML frontmatter block followed by Markdown
instructions:

```markdown
---
name: gloucester-weather
description: Give a concise weather briefing for Gloucester, UK.
---

# Gloucester weather

Use the weather tool without a location to respect the configured profile.
Report current conditions first, then mention rain or wind only when notable.
Keep the spoken response to two sentences.
```

- `name` must match the directory name; `description` is required and is what
   appears in the prompt index.
- Keep instructions task-focused and normally under a few hundred words.
   Skills are operational recipes, not background essays.
- A skill may include a `references/` directory for longer supporting material
   and a `scripts/` directory for deterministic helpers. The main skill should
   state when to read a reference or run a script; do not automatically load
   either.
- Do not put secrets in skills. Store configuration in `vivo.toml` or the
   environment, and have skills refer to named configuration instead.

## Invocation and behavior

- When a skill is loaded, treat its instructions as a task-specific extension
   of the system prompt for that request only. Base safety, voice, and tool
   constraints still win.
- Prefer an exact skill when one matches; otherwise follow the ordinary agent
   flow. Never force a skill solely because its name shares a word with the
   request.
- Announce only meaningful slow work in the spoken response. Do not read skill
   names, file paths, or procedural instructions aloud unless the user asks.
- If a skill is missing, malformed, or conflicts with the available tools,
   state the operational limitation briefly and continue with the best general
   approach.

## Maintenance

- Review skills after tool, model, endpoint, or workflow changes; retire stale
   ones rather than letting the index grow indefinitely.
- Add a skill after a repeatable task has demonstrated enough detail or failure
   modes to justify one. One-off requests should remain ordinary conversations.
- Test each skill's expected tool calls and output shape. Keep its description
   specific enough that the agent can select it without guessing.
