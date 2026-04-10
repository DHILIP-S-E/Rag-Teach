"""
LLM generation module using Google Gemini with grounded RAG prompt engineering.

The prompt template is designed to:
1. Ground the model strictly in retrieved context
2. Prevent hallucination by instructing "answer only from context"
3. Provide a clear fallback when context is insufficient
4. Support conversational memory by including chat history
"""

import google.generativeai as genai
from config import Config


# The system prompt establishes the assistant's behavior: a grounded
# document QA agent that only answers from provided context.
SYSTEM_PROMPT = """You are a precise document assistant. Your ONLY job is to answer questions using the provided context passages.

RULES:
1. Answer ONLY based on the information in the CONTEXT below.
2. If the context does not contain enough information to answer, say: "I don't have enough information in the uploaded documents to answer this question."
3. When answering, cite which source document and page the information came from.
4. Be concise and direct. Do not add information beyond what the context provides.
5. If the question is ambiguous, ask for clarification.
6. Maintain a professional, helpful tone."""


def build_context_block(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured context block
    that the LLM can easily parse and cite.
    """
    if not retrieved_chunks:
        return "No relevant documents found."

    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        score = chunk.get("combined_score", chunk.get("score", 0))
        text = chunk.get("text", "")
        parts.append(
            f"[Source {i}] {source} (Page {page}) [Relevance: {score:.2f}]\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def build_gemini_history(chat_history: list[dict] | None = None) -> list[dict]:
    """
    Convert chat history to Gemini's format.
    Gemini uses 'user' and 'model' roles (not 'assistant').
    """
    if not chat_history:
        return []

    gemini_history = []
    for msg in chat_history[-20:]:  # Keep last 10 exchanges
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append({"role": role, "parts": [msg["content"]]})
    return gemini_history


class Generator:
    """LLM-based answer generator using Google Gemini."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model_name = model or Config.LLM_MODEL
        api_key = api_key or Config.GEMINI_API_KEY
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT,
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        chat_history: list[dict] | None = None,
        temperature: float = 0.1,  # Low temp for factual grounding
    ) -> str:
        """
        Generate a grounded answer from retrieved context.
        Low temperature (0.1) reduces creative/hallucinated output.
        """
        context_block = build_context_block(retrieved_chunks)
        user_message = f"""CONTEXT:
{context_block}

QUESTION: {query}

Answer the question using ONLY the context above. Cite your sources."""

        try:
            # Start a chat with history for conversational continuity
            gemini_history = build_gemini_history(chat_history)
            chat = self.model.start_chat(history=gemini_history)

            response = chat.send_message(
                user_message,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1024,
                ),
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_stream(
        self,
        query: str,
        retrieved_chunks: list[dict],
        chat_history: list[dict] | None = None,
        temperature: float = 0.1,
    ):
        """
        Streaming version for real-time UI updates.
        Yields text chunks as they arrive from the Gemini API.
        """
        context_block = build_context_block(retrieved_chunks)
        user_message = f"""CONTEXT:
{context_block}

QUESTION: {query}

Answer the question using ONLY the context above. Cite your sources."""

        try:
            gemini_history = build_gemini_history(chat_history)
            chat = self.model.start_chat(history=gemini_history)

            response = chat.send_message(
                user_message,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=1024,
                ),
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error generating response: {str(e)}"
