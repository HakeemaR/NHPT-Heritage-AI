from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_architectural_style_classifier.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

IMAGE_SIZE = (224, 224)
MIN_CONFIDENCE_THRESHOLD = 0.60
MAX_HISTORY_MESSAGES = 8
RETRIEVAL_K = 4


st.set_page_config(
    page_title="NHPT Heritage AI",
    page_icon="🏛️",
    layout="wide",
)


# ============================================================
# RESOURCE LOADING
# ============================================================

@st.cache_resource
def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class-name file not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise ValueError("class_names.json must contain a JSON list.")

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "The number of classifier outputs does not match "
            "the number of class names."
        )

    return model, class_names


@st.cache_resource
def load_rag_components():
    index_file = VECTORSTORE_DIR / "index.faiss"
    metadata_file = VECTORSTORE_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            "FAISS vector-store files were not found. "
            "Run: python build_vectorstore.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.2,
    )

    return vector_store, llm


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "predicted_style": None,
        "confidence": None,
        "ranked_predictions": None,
        "uploaded_image_hash": None,
        "uploaded_image_bytes": None,
        "last_sources": [],
        "analysis_allowed": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis() -> None:
    st.session_state.messages = []
    st.session_state.predicted_style = None
    st.session_state.confidence = None
    st.session_state.ranked_predictions = None
    st.session_state.uploaded_image_hash = None
    st.session_state.uploaded_image_bytes = None
    st.session_state.last_sources = []
    st.session_state.analysis_allowed = False


# ============================================================
# IMAGE PROCESSING AND CLASSIFICATION
# ============================================================

def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    if image_array.shape != (224, 224, 3):
        raise ValueError(
            f"Unexpected preprocessed image shape: {image_array.shape}"
        )

    return np.expand_dims(image_array, axis=0)


def predict_style(model, class_names, image: Image.Image):
    image_array = prepare_image(image)
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_style = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    ranked_predictions = sorted(
        [
            (class_name, float(probability))
            for class_name, probability in zip(
                class_names,
                probabilities,
            )
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    return predicted_style, confidence, ranked_predictions


def get_confidence_status(confidence: float) -> tuple[str, str]:
    if confidence >= 0.90:
        return "Very High Confidence", "success"

    if confidence >= 0.75:
        return "High Confidence", "success"

    if confidence >= MIN_CONFIDENCE_THRESHOLD:
        return "Moderate Confidence", "warning"

    return "Low Confidence", "error"


# ============================================================
# RAG AND CONVERSATIONAL MEMORY
# ============================================================

def format_conversation_history(
    messages: list[dict],
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> str:
    recent_messages = messages[-max_messages:]

    if not recent_messages:
        return "No previous conversation."

    lines = []

    for message in recent_messages:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")

    return "\n".join(lines)


def retrieve_documents(vector_store, question: str, predicted_style: str):
    retrieval_query = (
        f"Predicted architectural style: {predicted_style}. "
        f"Question: {question}"
    )

    documents = vector_store.similarity_search(
        retrieval_query,
        k=RETRIEVAL_K,
    )

    return documents


def build_context(documents) -> tuple[str, list[dict]]:
    context_sections = []
    source_records = []

    for number, document in enumerate(documents, start=1):
        source_name = document.metadata.get("source", "Unknown source")
        style_name = document.metadata.get("style", "Unknown style")
        chunk_id = document.metadata.get("chunk_id", "Unknown")

        source_records.append(
            {
                "number": number,
                "source": source_name,
                "style": style_name,
                "chunk_id": chunk_id,
            }
        )

        context_sections.append(
            f"[Source {number}: {source_name}, chunk {chunk_id}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_sections), source_records


def generate_rag_answer(
    vector_store,
    llm,
    question: str,
    predicted_style: str,
    confidence: float,
    messages: list[dict],
):
    documents = retrieve_documents(
        vector_store,
        question,
        predicted_style,
    )

    if not documents:
        return (
            "I could not find enough relevant information in the "
            "heritage knowledge base to answer that question.",
            [],
        )

    context, _ = build_context(documents)
    conversation_history = format_conversation_history(messages)

    prompt = f"""
You are the conversational heritage assistant for the
National Heritage Preservation Trust.

Computer-vision structured output:
- Predicted style: {predicted_style}
- Confidence: {confidence * 100:.2f}%

Previous conversation:
{conversation_history}

Retrieved knowledge-base context:
{context}

Current user question:
{question}

Instructions:
1. Answer only from the retrieved context.
2. Use previous conversation history to understand follow-up questions.
3. Cite factual claims using [Source 1], [Source 2], and so on.
4. Do not claim that you directly inspected image details.
5. Do not invent dates, names, buildings, materials, or sources.
6. If the context is insufficient, clearly say so.
7. Keep the answer clear and suitable for a visitor or university student.
"""

    response = llm.invoke(prompt)
    answer = response.content.strip()

    return answer, documents


def generate_initial_explanation(
    vector_store,
    llm,
    predicted_style: str,
    confidence: float,
):
    initial_question = (
        f"Explain {predicted_style}. Include its historical background, "
        f"main visual characteristics, common materials, structural "
        f"features, famous examples, and how it differs from a similar style."
    )

    return generate_rag_answer(
        vector_store=vector_store,
        llm=llm,
        question=initial_question,
        predicted_style=predicted_style,
        confidence=confidence,
        messages=[],
    )


# ============================================================
# USER INTERFACE
# ============================================================

def display_header() -> None:
    st.title("🏛️ NHPT Heritage AI")

    st.markdown(
        """
        ### Architectural Heritage Recognition using Deep Learning and RAG

        Upload a building image to classify its architectural style.
        For supported predictions, the system retrieves relevant heritage
        knowledge and provides a cited, conversational explanation.
        """
    )

    st.caption(
        "Supported styles: Ancient Egyptian, Art Deco, Baroque, "
        "Bauhaus and Byzantine architecture."
    )


def display_prediction_results() -> None:
    predicted_style = st.session_state.predicted_style
    confidence = st.session_state.confidence
    ranked_predictions = st.session_state.ranked_predictions

    confidence_label, message_type = get_confidence_status(confidence)

    st.subheader("🏛️ Classification Result")

    if confidence >= MIN_CONFIDENCE_THRESHOLD:
        st.success(predicted_style)
    else:
        st.error("Unsupported or uncertain image")

    st.metric(
        label="Confidence",
        value=f"{confidence * 100:.2f}%",
    )

    if message_type == "success":
        st.success(f"🟢 {confidence_label}")

    elif message_type == "warning":
        st.warning(f"🟡 {confidence_label}")

    else:
        st.error(f"🔴 {confidence_label}")

        st.warning(
            "The uploaded image may not contain one of the supported "
            "architectural styles, or the visual evidence is too uncertain. "
            "The RAG explanation and chat have been disabled to prevent "
            "an unreliable or misleading response."
        )

    st.subheader("📊 Top Predictions")

    for class_name, probability in ranked_predictions[:3]:
        st.write(f"**{class_name}** — {probability * 100:.2f}%")
        st.progress(probability)


def display_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_sources() -> None:
    if not st.session_state.last_sources:
        return

    with st.expander("📚 View retrieved knowledge sources"):
        for number, document in enumerate(
            st.session_state.last_sources,
            start=1,
        ):
            source = document.metadata.get("source", "Unknown source")
            style = document.metadata.get("style", "Unknown style")
            chunk_id = document.metadata.get("chunk_id", "Unknown")

            st.markdown(
                f"**Source {number}: {source}**  \n"
                f"Style: {style}  \n"
                f"Chunk: {chunk_id}"
            )

            st.write(document.page_content)
            st.divider()


def process_uploaded_image(
    uploaded_file,
    model,
    class_names,
    vector_store,
    llm,
) -> None:
    image_bytes = uploaded_file.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    if image_hash == st.session_state.uploaded_image_hash:
        return

    try:
        image = Image.open(uploaded_file)
        image.verify()
        image = Image.open(uploaded_file).convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        st.error(
            "The uploaded file is not a valid image. "
            "Please upload a JPG, JPEG, PNG or WEBP image."
        )
        raise ValueError("Invalid image upload.") from error

    with st.spinner("Analysing architectural style..."):
        predicted_style, confidence, ranked_predictions = predict_style(
            model,
            class_names,
            image,
        )

    analysis_allowed = confidence >= MIN_CONFIDENCE_THRESHOLD

    if analysis_allowed:
        try:
            with st.spinner(
                "Retrieving heritage knowledge and generating explanation..."
            ):
                explanation, source_documents = generate_initial_explanation(
                    vector_store,
                    llm,
                    predicted_style,
                    confidence,
                )

            messages = [
                {
                    "role": "assistant",
                    "content": explanation,
                }
            ]

        except Exception as error:
            source_documents = []
            messages = [
                {
                    "role": "assistant",
                    "content": (
                        "The architectural style was classified, but the "
                        "knowledge assistant is temporarily unavailable. "
                        "Please confirm that Ollama is running and that "
                        f"the model '{OLLAMA_MODEL}' is installed."
                    ),
                }
            ]

            st.error(f"RAG/LLM error: {error}")

    else:
        source_documents = []
        messages = [
            {
                "role": "assistant",
                "content": (
                    "I could not confidently identify this image as one of "
                    "the five supported architectural styles. The explanation "
                    "and chat were disabled to prevent a misleading response."
                ),
            }
        ]

    st.session_state.uploaded_image_hash = image_hash
    st.session_state.uploaded_image_bytes = image_bytes
    st.session_state.predicted_style = predicted_style
    st.session_state.confidence = confidence
    st.session_state.ranked_predictions = ranked_predictions
    st.session_state.last_sources = source_documents
    st.session_state.messages = messages
    st.session_state.analysis_allowed = analysis_allowed


def main() -> None:
    initialize_session_state()
    display_header()

    try:
        model, class_names = load_classifier()
        vector_store, llm = load_rag_components()

    except Exception as error:
        st.error(f"Application setup error: {error}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an architectural image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload one clear image containing a building or architectural feature.",
    )

    if uploaded_file is not None:
        try:
            process_uploaded_image(
                uploaded_file,
                model,
                class_names,
                vector_store,
                llm,
            )
        except ValueError:
            st.stop()

    if st.session_state.predicted_style is None:
        st.info("Upload an image to begin.")
        return

    image_column, result_column = st.columns([1, 1])

    with image_column:
        st.subheader("🖼️ Uploaded Image")

        if st.session_state.uploaded_image_bytes:
            st.image(
                st.session_state.uploaded_image_bytes,
                use_container_width=True,
            )

    with result_column:
        display_prediction_results()

    st.divider()
    st.subheader("🤖 Conversational Heritage Assistant")

    st.caption(
        "The assistant explains the architectural style predicted by the "
        "computer-vision model. It does not independently inspect or verify "
        "specific visual details in the uploaded image."
    )

    display_chat_history()
    display_sources()

    if not st.session_state.analysis_allowed:
        st.info(
            "The conversational assistant is unavailable because the "
            f"classification confidence is below "
            f"{MIN_CONFIDENCE_THRESHOLD * 100:.0f}%."
        )
        user_question = None

    else:
        user_question = st.chat_input(
            "Ask a follow-up question about this architectural style"
        )

    if user_question:
        cleaned_question = user_question.strip()

        if not cleaned_question:
            st.warning("Please enter a question.")
        else:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": cleaned_question,
                }
            )

            with st.chat_message("user"):
                st.markdown(cleaned_question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner(
                        "Searching the heritage knowledge base..."
                    ):
                        answer, source_documents = generate_rag_answer(
                            vector_store=vector_store,
                            llm=llm,
                            question=cleaned_question,
                            predicted_style=(
                                st.session_state.predicted_style
                            ),
                            confidence=st.session_state.confidence,
                            messages=st.session_state.messages[:-1],
                        )

                    st.markdown(answer)

                except Exception as error:
                    answer = (
                        "I could not generate an answer because the local "
                        "language model is unavailable. Please check that "
                        "Ollama is running."
                    )
                    source_documents = []
                    st.error(f"RAG/LLM error: {error}")
                    st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            st.session_state.last_sources = source_documents
            st.rerun()

    if st.button("Reset image and conversation"):
        reset_analysis()
        st.rerun()


if __name__ == "__main__":
    main()