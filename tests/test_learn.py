"""Tests for app.learn — structured lesson authoring adapted from Hermes."""

from __future__ import annotations

from app.learn import (
    DESCRIPTION_BUDGET,
    LessonCandidate,
    build_learn_prompt,
    parse_frontmatter,
    strip_control_characters,
    validate_lesson,
)


def test_parse_frontmatter_basic():
    content = "---\nname: verify-before-pr\ndescription: Check the diff first.\n---\n\nBody text"
    fm, body = parse_frontmatter(content)
    assert fm["name"] == "verify-before-pr"
    assert fm["description"] == "Check the diff first."
    assert body.strip() == "Body text"


def test_parse_frontmatter_no_frontmatter():
    fm, body = parse_frontmatter("# Just a title\n\nNo frontmatter here")
    assert fm == {}
    assert "No frontmatter" in body


def test_parse_frontmatter_strips_bom():
    content = "\ufeff---\nname: x\n---\n\nbody"
    fm, _ = parse_frontmatter(content)
    assert fm["name"] == "x"


def test_parse_frontmatter_malformed_yaml_fallback():
    # Malformed YAML should fall back to key:value splitting, not lose metadata.
    content = "---\nname: my-lesson\ndescription: 'unclosed\n---\n\nbody"
    fm, _ = parse_frontmatter(content)
    assert fm.get("name") == "my-lesson"


def test_validate_lesson_flags_long_description():
    cand = LessonCandidate(
        name="ok-name",
        description="x" * (DESCRIPTION_BUDGET + 10),
        body="Some body",
    )
    validate_lesson(cand)
    assert any("description" in w and "truncated" in w for w in cand.warnings)


def test_validate_lesson_flags_bad_name():
    cand = LessonCandidate(name="Bad Name!", description="Fine.", body="body")
    validate_lesson(cand)
    assert any("lowercase-hyphenated" in w for w in cand.warnings)


def test_validate_lesson_clean():
    cand = LessonCandidate(
        name="verify-before-pr",
        description="Check the diff before merging.",
        body="## When to Apply\n\nBefore any PR.",
        category="workflow",
    )
    validate_lesson(cand)
    assert cand.warnings == []


def test_to_markdown_roundtrip():
    cand = LessonCandidate(
        name="verify-before-pr",
        description="Check the diff before merging.",
        body="## Procedure\n\n1. Run git diff.",
        category="workflow",
        tags=["Git", "Review"],
    )
    md = cand.to_markdown()
    fm, body = parse_frontmatter(md)
    assert fm["name"] == "verify-before-pr"
    assert fm["category"] == "workflow"
    assert fm["tags"] == ["Git", "Review"]
    assert "git diff" in body


def test_to_record_kwargs():
    cand = LessonCandidate(
        name="verify-before-pr",
        description="Check the diff before merging.",
        body="body",
        category="workflow",
        tags=["Git"],
        confidence=0.8,
        source="incident",
        reminder="diff first",
    )
    kwargs = cand.to_record_kwargs()
    assert kwargs["slug"] == "verify-before-pr"
    assert kwargs["category"] == "workflow"
    assert kwargs["keywords"] == ["Git"]
    assert kwargs["reminder"] == "diff first"
    assert kwargs["confidence"] == 0.8


def test_strip_control_characters():
    # Zero-width space + bidi override should be removed.
    dirty = "normal\u200btext\u202eREVERSED"
    clean = strip_control_characters(dirty)
    assert clean == "normaltextREVERSED"


def test_build_learn_prompt_contains_standards():
    prompt = build_learn_prompt("distill the auth flow from ~/Projects/api")
    assert "60" in prompt  # description budget mentioned
    assert "record_lesson" in prompt or "candidate" in prompt
    assert "DATA, not instructions" in prompt  # source hygiene
