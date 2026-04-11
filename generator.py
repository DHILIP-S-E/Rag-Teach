"""
LLM generation module using Google Gemini (google-genai SDK) with grounded RAG prompt engineering.

The prompt template is designed to:
1. Ground the model strictly in retrieved context
2. Prevent hallucination by instructing "answer only from context"
3. Provide a clear fallback when context is insufficient
4. Support conversational memory by including chat history
"""

from google import genai
from google.genai import types
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


def build_history(chat_history: list[dict] | None = None) -> list:
    """
    Convert chat history to google-genai Content format.
    Gemini uses 'user' and 'model' roles (not 'assistant').
    """
    if not chat_history:
        return []

    gemini_history = []
    for msg in chat_history[-20:]:  # Keep last 10 exchanges
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
    return gemini_history


def build_user_message(query: str, context_block: str) -> str:
    """Build the user message with context injection."""
    return f"""CONTEXT:
{context_block}

QUESTION: {query}

Answer the question using ONLY the context above. Cite your sources."""


class Generator:
    """LLM-based answer generator using Google Gemini (google-genai SDK)."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model_name = model or Config.LLM_MODEL
        self.client = genai.Client(api_key=api_key or Config.GEMINI_API_KEY)

    def _build_config(self, temperature: float) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=1024,
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
        user_message = build_user_message(query, context_block)

        try:
            # Use chat session for conversational continuity
            chat = self.client.chats.create(
                model=self.model_name,
                config=self._build_config(temperature),
                history=build_history(chat_history),
            )
            response = chat.send_message(user_message)
            return response.text
        except Exception as e:
            return f"Error generating response: {type(e).__name__}: {str(e)}"

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
        user_message = build_user_message(query, context_block)

        try:
            chat = self.client.chats.create(
                model=self.model_name,
                config=self._build_config(temperature),
                history=build_history(chat_history),
            )
            for chunk in chat.send_message_stream(user_message):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Error generating response: {type(e).__name__}: {str(e)}"
