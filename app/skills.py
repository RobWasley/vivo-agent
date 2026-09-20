"""Disk-backed, progressively disclosed agent skills."""
from __future__ import annotations

import re
import shutil
from pathlib import Path


DEFAULT_SKILLS_PATH = "data/skills"
SKILL_FILE = "SKILL.md"
MAX_SKILL_CHARS = 12_000
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillStore:
    """Validated skill files stored one per directory beneath a fixed root."""

    def __init__(self, path: str = DEFAULT_SKILLS_PATH):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_name(name: str) -> str:
        slug = str(name).strip().lower()
        if not _NAME_RE.fullmatch(slug):
            raise ValueError("skill name must use lowercase letters, numbers, and hyphens")
        return slug

    def _skill_path(self, name: str) -> Path:
        return self.path / self._validate_name(name) / SKILL_FILE

    @staticmethod
    def _parse(text: str) -> tuple[dict[str, str], str]:
        if not text.startswith("---\n"):
            raise ValueError("skill must start with YAML frontmatter")
        _, separator, body = text[4:].partition("\n---\n")
        if not separator:
            raise ValueError("skill frontmatter is not closed")

        metadata: dict[str, str] = {}
        for line in text[4:].split("\n---\n", 1)[0].splitlines():
            key, colon, value = line.partition(":")
            if not colon or key.strip() not in {"name", "description"}:
                raise ValueError("skill frontmatter requires only name and description")
            metadata[key.strip()] = value.strip().strip('"')
        if not metadata.get("name") or not metadata.get("description"):
            raise ValueError("skill frontmatter requires name and description")
        if not body.strip():
            raise ValueError("skill instructions cannot be empty")
        return metadata, body.strip()

    def list(self) -> list[dict[str, str]]:
        skills = []
        for skill_file in sorted(self.path.glob(f"*/{SKILL_FILE}")):
            try:
                metadata, _ = self._parse(skill_file.read_text(encoding="utf-8"))
                name = self._validate_name(metadata["name"])
                if name != skill_file.parent.name:
                    continue
                skills.append({"name": name, "description": metadata["description"]})
            except (OSError, ValueError):
                continue
        return skills

    def read(self, name: str) -> str:
        return self.details(name)["content"]

    def details(self, name: str) -> dict[str, str]:
        skill_file = self._skill_path(name)
        if not skill_file.is_file():
            raise FileNotFoundError(f"unknown skill: {name}")
        text = skill_file.read_text(encoding="utf-8")
        if len(text) > MAX_SKILL_CHARS:
            raise ValueError(f"skill exceeds {MAX_SKILL_CHARS} characters")
        metadata, instructions = self._parse(text)
        if self._validate_name(metadata["name"]) != skill_file.parent.name:
            raise ValueError("skill name does not match its directory")
        return {
            "name": metadata["name"],
            "description": metadata["description"],
            "instructions": instructions,
            "content": text.strip(),
        }

    def save(self, name: str, description: str, instructions: str) -> str:
        slug = self._validate_name(name)
        description = str(description).strip()
        instructions = str(instructions).strip()
        if not description:
            raise ValueError("skill description is required")
        if "\n" in description or len(description) > 240:
            raise ValueError("skill description must be one line of at most 240 characters")
        if not instructions:
            raise ValueError("skill instructions are required")
        content = f"---\nname: {slug}\ndescription: {description}\n---\n\n{instructions}\n"
        if len(content) > MAX_SKILL_CHARS:
            raise ValueError(f"skill exceeds {MAX_SKILL_CHARS} characters")
        skill_file = self._skill_path(slug)
        existed = skill_file.exists()
        skill_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = skill_file.with_suffix(".tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(skill_file)
        return "updated" if existed else "created"

    def delete(self, name: str) -> None:
        skill_dir = self._skill_path(name).parent
        if not skill_dir.is_dir():
            raise FileNotFoundError(f"unknown skill: {name}")
        shutil.rmtree(skill_dir)

    def index(self, limit: int = 20) -> str:
        entries = self.list()[:limit]
        if not entries:
            return ""
        summary = "; ".join(f"{item['name']}: {item['description']}" for item in entries)
        return "Available skills: " + summary + ". Use read_skill to load one when relevant."