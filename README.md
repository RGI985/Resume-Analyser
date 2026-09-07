# AI Resume Ranking & Voice PDF Agent

This project contains two Streamlit applications that use Ollama, FAISS, and PDF processing to deliver AI-powered document workflows.

## Included Apps

- `voice_app.py` voice mode
  - Voice-enabled PDF question-answering agent.
  - Upload a PDF, index its pages with embeddings, ask questions via a chat interface, and hear spoken answers.

- `main.py`
  - A simple placeholder entrypoint that prints a greeting.

- `model.py`
  - AI Resume Ranking app.
  - Upload PDF resumes and a job description, rank candidates using embeddings + FAISS, then analyze top matches with Ollama.

## Requirements

- Python 3.12+
- `streamlit`
- `requests`
- `fitz` (PyMuPDF)
- `faiss`
- `numpy`
- `pyttsx3`
- Local Ollama server running and accessible at `http://localhost:11434`

## Setup

1. Create and activate a Python virtual environment.
2. Install the required packages, for example:

```bash
pip install streamlit requests pymupdf faiss-cpu numpy pyttsx3
```

3. Start Ollama locally and ensure the embedding and generation APIs are reachable.

## Usage

### Run the voice-enabled PDF agent

```bash
streamlit run voice_app.py
```

### Use the resume ranking app

```bash
streamlit run model.py
```

## Notes

- The `voice_app.py` app extracts text from PDF pages, builds a FAISS index, and uses Ollama to answer questions from the most relevant page.
- The `model.py` app embeds resumes and the job description, ranks candidates by similarity, then asks Ollama to produce a candidate summary and score.
- The `main.py` file is a minimal example entrypoint.

## License

This repository does not specify a license. Add one if you plan to share or publish the project.
