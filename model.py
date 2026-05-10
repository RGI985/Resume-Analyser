import streamlit as st
import fitz  # PyMuPDF
import requests
import re
import faiss
import numpy as np

# 🔗 Ollama Config
OLLAMA_URL = "http://localhost:11434/api/generate"
EMBED_URL = "http://localhost:11434/api/embeddings"
LLM_MODEL = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

# 📄 Extract text from PDF
def extract_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 🔢 Get Embedding
def get_embedding(text):
    response = requests.post(EMBED_URL, json={
        "model": EMBED_MODEL,
        "prompt": text
    })
    return response.json()["embedding"]

# 🤖 Call Ollama LLM
def call_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

# 🧠 Analyze Candidate (LLM)
def analyze_candidate(jd, resume_text):
    prompt = f"""
    You are an AI recruiter.

    Job Description:
    {jd}

    Resume:
    {resume_text}

    Do ALL of the following:
    1. Extract candidate name
    2. List key skills
    3. Give a match score out of 100
    4. Give 2-line reason

    Format:
    Name:
    Skills:
    Score:
    Reason:
    """
    return call_ollama(prompt)

# 🔢 Extract score
def extract_score(text):
    match = re.search(r"Score[:\s]*(\d+)", text)
    return int(match.group(1)) if match else 0

# 🎨 Streamlit UI
st.set_page_config(page_title="AI Resume Ranker", layout="wide")

st.title("🚀 AI Resume Ranking Agent")
st.write("⚡ Power Product Manager hiring ")

# 📌 JD Input
jd = st.text_area("📄 Enter Job Description", height=150)

# 📂 Upload resumes
uploaded_files = st.file_uploader(
    "📂 Upload Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("⚡ Rank Candidates"):
    if not jd or not uploaded_files:
        st.warning("Please provide both JD and resumes")
    else:
        with st.spinner("🔄 Processing resumes with FAISS + AI..."):

            resume_texts = []
            resume_names = []
            embeddings = []

            # 📥 Step 1: Extract + Embed resumes
            for file in uploaded_files:
                text = extract_text(file)
                emb = get_embedding(text)

                resume_texts.append(text)
                resume_names.append(file.name)
                embeddings.append(emb)

            # 🔢 Convert to numpy
            vectors = np.array(embeddings).astype("float32")

            # 🧠 Create FAISS index
            dimension = vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(vectors)

            # 📌 Embed JD
            jd_embedding = get_embedding(jd)
            jd_vector = np.array([jd_embedding]).astype("float32")

            # 🔍 Search Top K
            k = min(5, len(uploaded_files))
            distances, indices = index.search(jd_vector, k)

            # 🤖 Analyze Top Candidates
            results = []

            for idx in indices[0]:
                resume_text = resume_texts[idx]
                name = resume_names[idx]

                analysis = analyze_candidate(jd, resume_text)
                score = extract_score(analysis)

                results.append({
                    "name": name,
                    "score": score,
                    "details": analysis
                })

        # 📊 Sort final ranking
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.success("✅ Ranking Complete!")

        # 🏆 Display results
        for i, res in enumerate(results):
            st.subheader(f"{i+1}. {res['name']} — {res['score']}%")
            st.write(res["details"])
            st.divider()


         
