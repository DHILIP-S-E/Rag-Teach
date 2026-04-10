"""
Streamlit UI for the RAG system.
Provides file upload, document processing, and a ChatGPT-style
conversational interface over uploaded documents.
"""

import streamlit as st
from config import Config
from ingestion import ingest_file
from chunking import chunk_documents
from embeddings import EmbeddingModel
from vector_store import VectorStore
from retriever import Retriever
from generator import Generator


# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)


# ── Initialize session state ─────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history": [],          # list of {"role": ..., "content": ...}
        "documents_processed": 0,
        "processing": False,
        "embedding_model": None,
        "vector_store": None,
        "retriever": None,
        "generator": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ── Lazy-initialize pipeline components ──────────────────────────────
@st.cache_resource
def get_embedding_model(provider: str, model_name: str) -> EmbeddingModel:
    """Cache the embedding model so it's loaded once across reruns."""
    return EmbeddingModel(provider=provider, model_name=model_name)


@st.cache_resource
def get_vector_store(collection: str) -> VectorStore:
    return VectorStore(collection_name=collection)


def get_pipeline():
    """Ensure all pipeline components are initialized."""
    if st.session_state.embedding_model is None:
        st.session_state.embedding_model = get_embedding_model(
            Config.EMBEDDING_PROVIDER, Config.EMBEDDING_MODEL
        )
    if st.session_state.vector_store is None:
        st.session_state.vector_store = get_vector_store(Config.QDRANT_COLLECTION)
    if st.session_state.retriever is None:
        st.session_state.retriever = Retriever(
            st.session_state.embedding_model,
            st.session_state.vector_store,
        )
    if st.session_state.generator is None:
        st.session_state.generator = Generator()


# ── Sidebar: settings + file upload ─────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Assistant")
    st.markdown("---")

    # Configuration warnings
    issues = Config.validate()
    if issues:
        for issue in issues:
            st.warning(f"⚠️ {issue}")

    # Re-ranking is always enabled
    use_reranking = True

    # File uploader
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🚀 Process Documents", use_container_width=True):
        get_pipeline()
        emb_model = st.session_state.embedding_model
        vs = st.session_state.vector_store

        # Ensure Qdrant collection exists
        vs.ensure_collection(vector_size=emb_model.dimension)

        total_chunks = 0
        progress = st.progress(0, text="Processing documents...")

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(
                (i) / len(uploaded_files),
                text=f"Processing: {uploaded_file.name}",
            )

            try:
                # Step 1: Extract text
                file_bytes = uploaded_file.read()
                documents = ingest_file(file_bytes, uploaded_file.name)

                if not documents:
                    st.warning(f"No text extracted from {uploaded_file.name}")
                    continue

                # Step 2: Chunk
                chunks = chunk_documents(documents)

                # Step 3: Embed
                texts = [c.text for c in chunks]
                embeddings = emb_model.embed_texts(texts)

                # Step 4: Store in Qdrant
                count = vs.upsert_documents(chunks, embeddings.tolist())
                total_chunks += count

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        progress.progress(1.0, text="Done!")
        st.session_state.documents_processed += total_chunks
        st.success(f"✅ Processed {total_chunks} chunks from {len(uploaded_files)} file(s)")

    # Collection info
    st.markdown("---")
    st.subheader("📊 Collection Info")
    try:
        get_pipeline()
        info = st.session_state.vector_store.get_collection_info()
        if info:
            st.metric("Stored Chunks", info.get("points_count", 0))
        else:
            st.info("No collection yet. Upload documents to start.")
    except Exception:
        st.info("Connect to Qdrant to see collection info.")

    # Clear collection button
    if st.button("🗑️ Clear Collection", use_container_width=True):
        try:
            get_pipeline()
            st.session_state.vector_store.delete_collection()
            st.session_state.documents_processed = 0
            st.success("Collection cleared.")
        except Exception as e:
            st.error(f"Error: {e}")

    # Clear chat button
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main chat area ───────────────────────────────────────────────────
st.title("💬 Chat with your Documents")

if not Config.GEMINI_API_KEY:
    st.info(
        "Set your `GEMINI_API_KEY` in the `.env` file to enable LLM responses. "
        "You can still upload and search documents with HuggingFace embeddings."
    )

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources if available
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for src in message["sources"]:
                    st.markdown(
                        f"**{src['source']}** (Page {src['page']}) — "
                        f"Relevance: {src.get('combined_score', src.get('score', 0)):.2f}"
                    )
                    st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                    st.markdown("---")

# Chat input
if query := st.chat_input("Ask a question about your documents..."):
    # Validate state
    if not Config.GEMINI_API_KEY:
        st.error("Please set your GEMINI_API_KEY in the .env file first.")
        st.stop()

    get_pipeline()

    # Check if collection exists before querying
    collection_info = st.session_state.vector_store.get_collection_info()
    if not collection_info or collection_info.get("points_count", 0) == 0:
        st.warning("No documents uploaded yet. Please upload and process documents first using the sidebar.")
        st.stop()

    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # Retrieve relevant chunks
            try:
                retrieved = st.session_state.retriever.retrieve(
                    query=query,
                    use_reranking=use_reranking,
                )
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                retrieved = []

        if not retrieved:
            response_text = (
                "I couldn't find any relevant information in the uploaded documents. "
                "Please make sure you've uploaded and processed documents first."
            )
            st.markdown(response_text)
            sources = []
        else:
            # Show sources in expander
            with st.expander("📎 Retrieved Sources", expanded=False):
                for i, chunk in enumerate(retrieved, 1):
                    score_val = chunk.get("combined_score", chunk.get("score", 0))
                    st.markdown(
                        f"**[{i}] {chunk['source']}** (Page {chunk['page']}) — "
                        f"Relevance: {score_val:.2f}"
                    )
                    st.text(
                        chunk["text"][:300] + "..."
                        if len(chunk["text"]) > 300
                        else chunk["text"]
                    )
                    st.markdown("---")

            # Stream the LLM response
            # Build chat history for context (exclude the sources metadata)
            llm_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history[:-1]  # exclude current query
            ]

            response_text = st.write_stream(
                st.session_state.generator.generate_stream(
                    query=query,
                    retrieved_chunks=retrieved,
                    chat_history=llm_history,
                )
            )
            sources = retrieved

    # Save assistant response to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response_text,
        "sources": sources,
    })
