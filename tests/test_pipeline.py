
from app.db import Database
from app.retrieval.pipeline import IndexingPipeline, file_content_hash
from tests.conftest import FakeRetriever


def test_file_content_hash(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    h1 = file_content_hash(f)
    assert h1 and len(h1) == 64  # sha256 hex
    f.write_text("world")
    h2 = file_content_hash(f)
    assert h1 != h2


def test_pipeline_indexes_and_skips(tmp_path):
    f1 = tmp_path / "a.py"
    f1.write_text("print('hello')\n" * 20)
    f2 = tmp_path / "b.md"
    f2.write_text("# B\n\ntext here\n" * 20)

    db = Database(str(tmp_path / "test.db"))
    retriever = FakeRetriever()
    pipeline = IndexingPipeline(retriever, db, chunk_size=200, chunk_overlap=0)

    # First index — both files should be indexed.
    stats = pipeline.index_source(str(tmp_path))
    assert stats.indexed == 2
    assert stats.skipped == 0
    assert stats.chunks_added > 0

    # Second index — both should be skipped (unchanged).
    stats2 = pipeline.index_source(str(tmp_path))
    assert stats2.indexed == 0
    assert stats2.skipped == 2

    # Modify one file — only that one should be re-indexed.
    f1.write_text("print('changed')\n" * 20)
    stats3 = pipeline.index_source(str(tmp_path))
    assert stats3.indexed == 1
    assert stats3.skipped == 1

    db.close()


def test_pipeline_force(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("print('hello')\n" * 20)

    db = Database(str(tmp_path / "test.db"))
    retriever = FakeRetriever()
    pipeline = IndexingPipeline(retriever, db, chunk_size=200, chunk_overlap=0)

    pipeline.index_source(str(tmp_path))
    stats = pipeline.index_source(str(tmp_path), force=True)
    assert stats.indexed == 1
    assert stats.skipped == 0
    db.close()


def test_pipeline_remove_file(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("print('hello')\n" * 20)

    db = Database(str(tmp_path / "test.db"))
    retriever = FakeRetriever()
    pipeline = IndexingPipeline(retriever, db, chunk_size=200, chunk_overlap=0)

    pipeline.index_source(str(tmp_path))
    assert db.file_hash_count(str(tmp_path)) == 1

    pipeline.remove_file(str(f), str(tmp_path))
    assert db.file_hash_count(str(tmp_path)) == 0
    db.close()
