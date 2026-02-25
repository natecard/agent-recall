from __future__ import annotations

from agent_recall.storage.base import Storage


def replicate_chunks_to(source: Storage, target: Storage) -> int:
    """Replicate all chunks from source storage to target storage.

    This function copies all chunks from a source Storage to a target Storage,
    preserving embeddings and metadata. It serves as the primitive for
    'export to cloud' features.

    Args:
        source: The source Storage to replicate chunks from.
        target: The target Storage to replicate chunks to.

    Returns:
        The number of chunks replicated.
    """
    chunks = source.list_chunks()
    for chunk in chunks:
        target.store_chunk(chunk)
        if chunk.embedding is not None:
            target.index_chunk_embedding(chunk.id, chunk.embedding)
    return len(chunks)
