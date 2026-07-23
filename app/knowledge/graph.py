"""Knowledge graph — LLM-based entity and relationship extraction.

Extracts entities (concepts, technologies, people, projects) and relationships
from indexed content, stores them in the `entities` and `relationships` SQLite
tables. The LLM does the extraction; SQLite stores the graph.

The extraction prompt asks the LLM to return structured JSON:
  {
    "entities": [{"name": "...", "type": "...", "description": "..."}],
    "relationships": [{"source": "...", "relation": "...", "target": "..."}]
  }

This is intentionally simple. The graph grows as content is indexed or as the
user explicitly runs `maos graph extract`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.db import Database
from app.log import get_logger
from app.models.base import Model
from app.models.types import Message

log = get_logger(__name__)

_EXTRACT_SYSTEM = """/no_think
You are a knowledge graph extractor. Given a text passage, extract entities and relationships as JSON.

Do NOT generate any reasoning, thinking, or explanation. Output the JSON object immediately.

Entity types: concept, technology, person, project, framework, paper, tool, language, organization, file, function.

Relationship examples: uses, implements, depends_on, relates_to, authored_by, part_of, affects, alternative_to, extends.

Return ONLY valid JSON, no prose, no thinking:
{
  "entities": [{"name": "EntityName", "type": "concept", "description": "one line"}],
  "relationships": [{"source": "EntityName", "relation": "uses", "target": "OtherEntity"}]
}

Keep entity names short and canonical (e.g. "MLX" not "the MLX framework"). Extract at most 10 entities and 15 relationships per passage."""


class KnowledgeGraph:
    """Manages entity/relationship extraction and storage."""

    def __init__(self, db: Database, planner: Model | None = None):
        self.db = db
        self.planner = planner

    def extract_from_text(self, text: str, source: str = "") -> dict[str, Any]:
        """Extract entities and relationships from a text passage using the LLM."""
        if not self.planner:
            return {"entities": [], "relationships": [], "error": "no planner available"}
        if len(text.strip()) < 50:
            return {"entities": [], "relationships": []}

        resp = self.planner.chat([
            Message("system", _EXTRACT_SYSTEM),
            Message("user", f"Extract from this passage:\n\n{text[:2000]}"),
        ])

        parsed = self._parse_json_response(resp.text)
        if parsed:
            self._store_extraction(parsed, source)
            return parsed
        return {"entities": [], "relationships": []}

    def extract_from_retriever(self, retriever, query: str, top_k: int = 5) -> dict[str, Any]:
        """Retrieve passages for a query, then extract entities from each."""
        results = retriever.search(query, top_k=top_k)
        total = {"entities": [], "relationships": []}
        for r in results:
            extracted = self.extract_from_text(r.text, source=r.source)
            if extracted:
                total["entities"].extend(extracted.get("entities", []))
                total["relationships"].extend(extracted.get("relationships", []))
        # Deduplicate.
        seen_entities = set()
        unique_entities = []
        for e in total["entities"]:
            key = e["name"].lower()
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(e)
        total["entities"] = unique_entities
        return total

    def _parse_json_response(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from an LLM response that may have surrounding text, code blocks, or reasoning traces."""
        # Strip reasoning traces from models like Nemotron that emit <think>...</think>.
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Strip markdown code fences if present.
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\s*```\s*$', '', cleaned)
        # Try parsing the whole thing first (most reliable if the model cooperates).
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Try to find the outermost JSON object.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        # Reasoning models (e.g. Nemotron) may emit prose before the JSON.
        # Try each "{" as a potential start, looking for valid JSON with our keys.
        for i, ch in enumerate(cleaned):
            if ch != "{":
                continue
            for j in range(len(cleaned), i, -1):
                if cleaned[j - 1] != "}":
                    continue
                try:
                    candidate = json.loads(cleaned[i:j])
                    if isinstance(candidate, dict) and ("entities" in candidate or "relationships" in candidate):
                        return candidate
                except json.JSONDecodeError:
                    continue
        log.warning("graph.parse_failed", text_preview=text[:200])
        return None

    def _store_extraction(self, parsed: dict[str, Any], source: str) -> None:
        """Store extracted entities and relationships in the database."""
        for entity in parsed.get("entities", []):
            name = entity.get("name", "").strip()
            etype = entity.get("type", "concept").strip()
            desc = entity.get("description", "").strip()
            if name:
                self.db.get_or_create_entity(name, etype, description=desc, source=source)

        for rel in parsed.get("relationships", []):
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            relation = rel.get("relation", "relates_to").strip()
            if src and tgt and relation:
                self.db.add_relationship(src, relation, tgt, source_ref=source)

        n_entities = len(parsed.get("entities", []))
        n_rels = len(parsed.get("relationships", []))
        log.info("graph.stored", entities=n_entities, relationships=n_rels, source=source)

    def search(self, name: str) -> list[dict[str, Any]]:
        """Search entities by name, returning their relationships."""
        entities = self.db.search_entities(name)
        results = []
        for e in entities:
            rels = self.db.relationships_for_entity(e["name"])
            results.append({
                "name": e["name"],
                "type": e["type"],
                "description": e["description"],
                "source": e["source"],
                "relationships": [
                    {"relation": r["relation"], "target": r["target"], "type": r["target_type"]}
                    for r in rels
                ],
            })
        return results

    def stats(self) -> dict[str, int]:
        return {
            "entities": self.db.entity_count(),
            "relationships": self.db.relationship_count(),
        }
