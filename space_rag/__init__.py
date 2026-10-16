"""space_rag: a tiny, readable Retrieval-Augmented Generation (RAG) pipeline.

Heavy dependencies (torch, transformers, sentence-transformers) are imported
lazily so that the core logic can be tested with only numpy installed.
"""

__version__ = "0.1.0"
