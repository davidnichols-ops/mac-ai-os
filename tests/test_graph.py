from app.db import Database
from app.knowledge.graph import KnowledgeGraph
from tests.conftest import FakeModel


def test_entity_storage(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    id1 = db.get_or_create_entity("MLX", "framework", description="Apple ML framework")
    id2 = db.get_or_create_entity("MLX", "framework")  # dedup
    assert id1 == id2
    assert db.entity_count() == 1
    db.close()


def test_relationship_storage(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.add_relationship("MLX", "uses", "Metal", source_type="framework", target_type="framework")
    db.add_relationship("MLX", "uses", "Metal")  # dedup (UNIQUE constraint)
    assert db.relationship_count() == 1
    assert db.entity_count() == 2

    rels = db.relationships_for_entity("MLX")
    assert len(rels) >= 1
    found = [r for r in rels if r["target"] == "Metal"]
    assert found
    db.close()


def test_graph_extract_from_text(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    planner = FakeModel()
    # Make the fake planner return JSON.
    planner._json_response = (
        '{"entities": [{"name": "MLX", "type": "framework", "description": "ML on Apple Silicon"}],'
        ' "relationships": [{"source": "MLX", "relation": "uses", "target": "Metal"}]}'
    )

    def chat_override(messages, **kwargs):
        from app.models.types import ModelResponse
        return ModelResponse(text=planner._json_response, model="fake", provider="fake")

    planner.chat = chat_override

    graph = KnowledgeGraph(db, planner=planner)
    result = graph.extract_from_text(
        "MLX is an array framework for machine learning on Apple Silicon. "
        "It uses Metal for GPU compute and benefits from unified memory.",
        source="test.md",
    )

    assert len(result.get("entities", [])) == 1
    assert result["entities"][0]["name"] == "MLX"
    assert db.entity_count() == 2  # MLX + Metal
    assert db.relationship_count() == 1
    db.close()


def test_graph_search(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.add_relationship("MLX", "uses", "Metal", source_type="framework", target_type="framework")
    db.add_relationship("MLX", "part_of", "Apple", source_type="framework", target_type="organization")

    graph = KnowledgeGraph(db)
    results = graph.search("MLX")
    assert len(results) == 1
    assert results[0]["name"] == "MLX"
    assert len(results[0]["relationships"]) >= 2
    db.close()


def test_graph_stats(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    db.add_relationship("A", "relates_to", "B")
    graph = KnowledgeGraph(db)
    stats = graph.stats()
    assert stats["entities"] == 2
    assert stats["relationships"] == 1
    db.close()
