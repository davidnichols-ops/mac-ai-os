"""`maos graph` — knowledge graph commands."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_markdown, print_panel

graph_app = typer.Typer(name="graph", help="Knowledge graph commands.", no_args_is_help=True)


@graph_app.command("stats")
def stats() -> None:
    """Show knowledge graph statistics."""
    rt = build_runtime()
    s = rt.graph.stats()
    print_panel(
        f"Entities:       {s['entities']}\n"
        f"Relationships:  {s['relationships']}",
        title="Knowledge Graph",
    )


@graph_app.command("search")
def search(
    name: str = typer.Argument(..., help="Entity name to search for"),
) -> None:
    """Search for an entity and show its relationships."""
    rt = build_runtime()
    results = rt.graph.search(name)
    if not results:
        typer.echo(f"No entities matching '{name}'.")
        return

    for entity in results:
        lines = [
            f"**{entity['name']}** ({entity['type']})",
        ]
        if entity.get("description"):
            lines.append(f"  {entity['description']}")
        if entity.get("source"):
            lines.append(f"  source: {entity['source']}")
        if entity["relationships"]:
            lines.append("  relationships:")
            for rel in entity["relationships"]:
                lines.append(f"    {rel['relation']} -> {rel['target']} ({rel['type']})")
        else:
            lines.append("  (no relationships)")
        print_markdown("\n".join(lines))


@graph_app.command("extract")
def extract(
    query: str = typer.Argument(..., help="Topic to extract entities for (retrieves from index)"),
) -> None:
    """Extract entities and relationships from indexed content about a topic."""
    rt = build_runtime()
    result = rt.graph.extract_from_retriever(rt.retriever, query, top_k=5)

    lines = [
        f"Extracted {len(result.get('entities', []))} entities, "
        f"{len(result.get('relationships', []))} relationships.",
        "",
        "Entities:",
    ]
    for e in result.get("entities", []):
        lines.append(f"  {e['name']} ({e.get('type', 'concept')})")
    lines.append("")
    lines.append("Relationships:")
    for r in result.get("relationships", []):
        lines.append(f"  {r['source']} --{r['relation']}--> {r['target']}")

    print_panel("\n".join(lines), title="Knowledge Graph Extraction")
    typer.echo(f"\nGraph now has {rt.graph.stats()['entities']} entities, "
               f"{rt.graph.stats()['relationships']} relationships.")


@graph_app.command("add")
def add(
    source: str = typer.Argument(..., help="Source entity name"),
    relation: str = typer.Argument(..., help="Relationship type (uses, depends_on, etc.)"),
    target: str = typer.Argument(..., help="Target entity name"),
) -> None:
    """Manually add a relationship to the knowledge graph."""
    rt = build_runtime()
    rt.db.add_relationship(source, relation, target)
    typer.echo(f"Added: {source} --{relation}--> {target}")
