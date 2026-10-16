# 🚀 Space RAG — a tiny RAG pipeline with a small local LLM

A minimal, readable **Retrieval-Augmented Generation** project that answers questions about
space exploration (Apollo, Artemis, Mars rovers, the ISS, JWST, Voyager, ISRO, reusable rockets)
using only **local, small models** — no API keys, no cloud, no GPU required.

| Component        | Choice                                    | Size      |
|------------------|-------------------------------------------|-----------|
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2`  | ~90 MB    |
| Generator (LLM)  | `Qwen/Qwen2.5-0.5B-Instruct`              | ~1 GB     |
| Vector store     | plain NumPy cosine similarity             | 0 deps    |

## How it works

```mermaid
flowchart LR
    A[Markdown docs in data/] --> B[Chunker<br/>120-word windows, 30 overlap]
    B --> C[Embedder<br/>MiniLM]
    C --> D[(Vector store<br/>NumPy)]
    Q[Question] --> E[Embed query]
    E --> D
    D -- top-k chunks --> F[Prompt builder<br/>'answer ONLY from context']
    F --> G[Qwen2.5-0.5B-Instruct]
    G --> H[Answer + cited sources]
```

1. **Ingest** – documents are split into overlapping chunks, embedded, and saved to `index/`.
2. **Retrieve** – the question is embedded and the `top_k` most similar chunks are found.
3. **Guardrail** – if even the best chunk is below `min_score` similarity, the system answers
   *"I don't know"* instead of letting a tiny model guess.
4. **Generate** – the chunks are placed in a grounded prompt and the small LLM writes a short,
   cited answer.

## Quick start

```bash
git clone <your-repo-url> space-rag && cd space-rag
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m space_rag ingest                            # build the index (first run downloads the embedding model)
python -m space_rag ask "Which Indian mission landed near the Moon's south pole?"
python -m space_rag chat                              # interactive mode
```

The first `ask` downloads the LLM (~1 GB) once; later runs use the local Hugging Face cache.
On a laptop CPU expect a few seconds per answer.

**No LLM yet?** Use retrieval-only mode to see exactly which passages are found:

```bash
python -m space_rag ask "How big is the JWST mirror?" --no-llm
```

### Example questions

- Who was the first person to walk on the Moon?
- Which mission made India the first nation to reach Mars orbit on its first attempt?
- How far from Earth does the James Webb Space Telescope orbit?
- Which spacecraft is the only one to have visited Uranus and Neptune?
- When did a Falcon 9 booster first land after an orbital launch?
- What is the capital of France? *(off-topic → the system should say it doesn't know)*

## Configuration

Everything lives in [`space_rag/config.py`](space_rag/config.py) and can be overridden with
environment variables:

| Variable              | Default                                   | Meaning                                  |
|-----------------------|-------------------------------------------|------------------------------------------|
| `RAG_LLM_MODEL`       | `Qwen/Qwen2.5-0.5B-Instruct`              | Any Hugging Face chat model              |
| `RAG_EMBED_MODEL`     | `sentence-transformers/all-MiniLM-L6-v2`  | Any sentence-transformers model          |
| `RAG_DOCS_DIR`        | `./data`                                  | Folder of `.md` / `.txt` documents       |
| `RAG_INDEX_DIR`       | `./index`                                 | Where the vector index is stored         |
| `RAG_CHUNK_WORDS`     | `120`                                     | Chunk size in words                      |
| `RAG_CHUNK_OVERLAP`   | `30`                                      | Words shared between neighbouring chunks |
| `RAG_TOP_K`           | `3`                                       | Passages given to the LLM                |
| `RAG_MIN_SCORE`       | `0.2`                                     | "I don't know" threshold (cosine sim.)   |
| `RAG_MAX_NEW_TOKENS`  | `256`                                     | Max length of the answer                 |

Want better answers and have a bit more RAM? Try a larger model:

```bash
RAG_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct python -m space_rag chat
```

## Use your own topic

The prompt is topic-agnostic, so switching topics is just swapping the documents:

1. Delete the files in `data/` and add your own `.md` or `.txt` files
   (start each file with a `# Title` line — it is used as the source name).
2. Run `python -m space_rag ingest`.

## Project layout

```
space-rag/
├── data/                  # knowledge base (8 short markdown documents)
├── space_rag/
│   ├── config.py          # settings + env overrides
│   ├── chunking.py        # load documents, split into overlapping chunks
│   ├── embeddings.py      # sentence-transformers wrapper
│   ├── vector_store.py    # NumPy cosine-similarity store + save/load
│   ├── llm.py             # small local LLM wrapper (transformers)
│   ├── pipeline.py        # retrieve → guardrail → prompt → generate
│   └── cli.py             # `ingest`, `ask`, `chat`
├── tests/                 # 25 tests, no model downloads needed
└── .github/workflows/ci.yml
```

## Tests

The tests use tiny fake embedder/LLM classes, so they run in under a second and need only
`numpy` and `pytest`:

```bash
pip install numpy pytest
pytest -q
```

## Limitations & ideas

- A 0.5B-parameter model is fast but can still misread or over-compress context. Citations
  (`[1]`, `[2]`) and the printed sources let you verify every answer.
- Brute-force NumPy search is ideal up to a few thousand chunks; for more, swap
  `VectorStore` for FAISS, Chroma or Qdrant — the pipeline only calls `.search()`.
- The sample facts were written for demonstration and spaceflight news moves quickly
  (e.g. Artemis schedules), so verify anything important against NASA / ISRO sources.
- Ideas: add a re-ranker, hybrid BM25 + vector search, PDF loading, a Gradio/Streamlit UI,
  or an evaluation set of question/answer pairs.

## License

MIT — see [LICENSE](LICENSE). Replace `<your name>` in the license with your own.
