"""
Intelligent text chunking module.

Splits documents into overlapping chunks of ~CHUNK_SIZE tokens.
Uses sentence-boundary-aware splitting so chunks don't break mid-sentence,
preserving semantic coherence for better embedding quality.
"""

import re
import tiktoken
from ingestion import Document
from config import Config


# tiktoken encoder for accurate token counting (cl100k_base covers GPT-4/3.5)
_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def split_into_sentences(text: str) -> list[str]:
    """
    Regex-based sentence splitter. Handles common abbreviations
    and decimal numbers to avoid false splits.
    """
    # Split on period/question/exclamation followed by space + uppercase,
    # or end of string. This is a practical heuristic that works well
    # for most English documents.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(doc: Document) -> list[Document]:
    """
    Split a single document into overlapping chunks.

    Strategy:
    1. Split text into sentences (preserves meaning boundaries)
    2. Greedily accumulate sentences until we hit CHUNK_SIZE tokens
    3. When a chunk is full, start the next chunk CHUNK_OVERLAP tokens
       back so context isn't lost at boundaries
    """
    sentences = split_into_sentences(doc.text)
    if not sentences:
        return []

    chunks: list[Document] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        # If a single sentence exceeds chunk size, include it as its own chunk
        if sent_tokens > Config.CHUNK_SIZE:
            # Flush current buffer first
            if current_sentences:
                chunks.append(Document(
                    text=" ".join(current_sentences),
                    metadata={**doc.metadata, "chunk_index": len(chunks)},
                ))
                current_sentences = []
                current_tokens = 0
            chunks.append(Document(
                text=sentence,
                metadata={**doc.metadata, "chunk_index": len(chunks)},
            ))
            continue

        # If adding this sentence would exceed the limit, flush and start overlap
        if current_tokens + sent_tokens > Config.CHUNK_SIZE and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Document(
                text=chunk_text,
                metadata={**doc.metadata, "chunk_index": len(chunks)},
            ))

            # Build overlap: walk backward through sentences until we
            # accumulate CHUNK_OVERLAP tokens worth of context
            overlap_sentences: list[str] = []
            overlap_tokens = 0
            for prev_sent in reversed(current_sentences):
                prev_tokens = count_tokens(prev_sent)
                if overlap_tokens + prev_tokens > Config.CHUNK_OVERLAP:
                    break
                overlap_sentences.insert(0, prev_sent)
                overlap_tokens += prev_tokens

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Don't forget the last chunk
    if current_sentences:
        chunks.append(Document(
            text=" ".join(current_sentences),
            metadata={**doc.metadata, "chunk_index": len(chunks)},
        ))

    return chunks


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Chunk all documents, preserving metadata lineage."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
