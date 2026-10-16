"""Command line interface:  python -m space_rag {ingest,ask,chat} ..."""

from __future__ import annotations

import argparse
import textwrap

from .config import Config
from .embeddings import Embedder
from .llm import SmallLLM
from .pipeline import Answer, RAGPipeline, build_index
from .vector_store import VectorStore


def _load_or_build_store(cfg: Config, embedder: Embedder, rebuild: bool = False) -> VectorStore:
    if not rebuild and VectorStore.exists(cfg.index_dir):
        store = VectorStore.load(cfg.index_dir)
        if store.meta.get("embed_model") == cfg.embed_model:
            return store
        print("Index was built with a different embedding model - rebuilding.")

    print(f"Building index from {cfg.docs_dir} ...")
    store = build_index(cfg.docs_dir, embedder, cfg.chunk_words, cfg.chunk_overlap)
    store.save(cfg.index_dir, meta={"embed_model": cfg.embed_model})
    print(f"Indexed {len(store)} chunks -> {cfg.index_dir}")
    return store


def _make_pipeline(cfg: Config, use_llm: bool, top_k: int | None, rebuild: bool = False) -> RAGPipeline:
    embedder = Embedder(cfg.embed_model)
    store = _load_or_build_store(cfg, embedder, rebuild)
    llm = SmallLLM(cfg.llm_model, cfg.max_new_tokens) if use_llm else None
    return RAGPipeline(embedder, store, llm, top_k or cfg.top_k, cfg.min_score)


def _print_answer(ans: Answer, show_passages: bool) -> None:
    if ans.text:
        print("\nAnswer:\n" + textwrap.fill(ans.text, 88, replace_whitespace=False))
    print("\nSources:")
    for i, h in enumerate(ans.hits, 1):
        print(f"  [{i}] {h.chunk.title} ({h.chunk.source}) - similarity {h.score:.2f}")
        if show_passages:
            print(textwrap.indent(textwrap.fill(h.chunk.text, 86), "      "))
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="space_rag", description="Space-exploration RAG with a small local LLM")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="(Re)build the vector index from the documents folder")

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question")
    p_ask.add_argument("--no-llm", action="store_true", help="retrieval only: show matching passages, skip generation")
    p_ask.add_argument("--top-k", type=int, default=None)

    p_chat = sub.add_parser("chat", help="Interactive question loop")
    p_chat.add_argument("--no-llm", action="store_true")
    p_chat.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args(argv)
    cfg = Config.from_env()

    if args.command == "ingest":
        _load_or_build_store(cfg, Embedder(cfg.embed_model), rebuild=True)
        return

    use_llm = not args.no_llm
    pipeline = _make_pipeline(cfg, use_llm, args.top_k)

    if args.command == "ask":
        _print_answer(pipeline.ask(args.question), show_passages=not use_llm)
        return

    print("Space RAG chat - type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            _print_answer(pipeline.ask(question), show_passages=not use_llm)


if __name__ == "__main__":
    main()
