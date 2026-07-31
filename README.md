# NHPT Heritage AI

An AI-powered heritage assistant developed for the National Heritage Preservation Trust (NHPT).

The system combines **Computer Vision**, **Retrieval-Augmented Generation (RAG)**, and a **Large Language Model (Llama 3.2)** to classify architectural styles and provide contextual heritage explanations through an interactive conversational interface.

---

## Features

- Architectural style classification using EfficientNetB0
- 92.76% test accuracy
- Confidence scores and Top-3 predictions
- Grad-CAM visualisations
- Retrieval-Augmented Generation (RAG)
- LangChain integration
- FAISS vector database
- Local Llama 3.2 via Ollama
- Multi-turn conversational memory
- Source citations
- Low-confidence detection
- Hallucination mitigation
- Streamlit web application

---

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- Streamlit
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- Llama 3.2
- Ollama
- Pillow
- NumPy

---

## System Architecture

*(Insert your architecture diagram here after exporting it.)*

---

## Project Structure

```text
app.py                     Streamlit application
models/                    Trained model
knowledge_base/            Heritage documents
vectorstore/               FAISS index
notebooks/                 Training notebook
assets/                    Screenshots and diagrams
```

---

## Dataset

Architectural Styles Dataset (Kaggle)

Classes used:

- Ancient Egyptian
- Art Deco
- Baroque
- Bauhaus
- Byzantine

---

## Model Performance

| Metric | Value |
|--------|------:|
| Test Accuracy | 92.76% |
| Precision | 93.82% |
| Recall | 92.74% |
| F1-score | 93.05% |

---

## Running the Application

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start Ollama

```bash
ollama serve
```

### Run Streamlit

```bash
streamlit run app.py
```

---

## Author

Hakeema Rizan

University Coursework – Artificial Intelligence for Heritage Applications