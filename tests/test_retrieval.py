from app.retrieval.chunking import chunk_text
from app.retrieval.indexers import build_indexers, dispatch
from app.retrieval.indexers.code import CodeIndexer
from app.retrieval.indexers.markdown import MarkdownIndexer


def test_chunk_text_overlap():
    text = "abcdefghij" * 50
    chunks = chunk_text(text, source="x", chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.source == "x" for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("", source="x") == []


def test_markdown_indexer(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# A\n\nbody one\n\n# B\n\nbody two\n")
    idx = MarkdownIndexer(chunk_size=800, chunk_overlap=0)
    chunks = idx.index(str(f))
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["kind"] == "markdown"


def test_code_indexer(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("def f():\n    return 1\n" * 20)
    idx = CodeIndexer(chunk_size=200, chunk_overlap=0)
    chunks = idx.index(str(f))
    assert chunks
    assert chunks[0]["metadata"]["language"] == "python"


def test_folder_indexer(tmp_path):
    (tmp_path / "a.py").write_text("print('a')\n" * 30)
    (tmp_path / "b.md").write_text("# B\n\ntext\n" * 30)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("git stuff")  # should be skipped
    indexers = build_indexers(chunk_size=200, chunk_overlap=0)
    chunks = dispatch(str(tmp_path), indexers)
    sources = {c["metadata"]["source"] for c in chunks}
    assert any("a.py" in s for s in sources)
    assert any("b.md" in s for s in sources)
    assert not any(".git" in s for s in sources)
