# 🧠 JobFit AI – Resume Matcher

JobFit AI is an AI-powered Resume–Job Description (JD) matching system that evaluates candidate suitability using semantic search, Retrieval-Augmented Generation (RAG), and local Large Language Models (LLMs). The system analyzes resumes against job descriptions, identifies skill gaps, generates ATS recommendations, and produces detailed visual reports — fully offline with zero API cost.

Built as an intelligent recruitment assistance system focused on semantic understanding rather than keyword-only matching.

---

# 🚀 Features

- 📄 Resume & JD Analysis
  - Upload resumes and job descriptions in PDF/text format
  - Extracts and processes textual content automatically

- 🔍 Semantic Resume Matching
  - Uses `all-MiniLM-L6-v2` embeddings for contextual similarity
  - Computes intelligent match scores beyond keyword matching

- ⚡ Fast Vector Search using FAISS
  - Retrieves top relevant resume and JD sections
  - Improves contextual comparison using vector similarity

- 🧠 Retrieval-Augmented Generation (RAG)
  - Overlapping chunking pipeline for better context retrieval
  - Supplies only relevant sections to the LLM

- 🤖 Local LLM Integration
  - Uses `Qwen2.5:3B` via Ollama
  - Fully offline and zero cloud/API dependency

- 📊 ATS & Skill Gap Analysis
  - Generates:
    - Match scores
    - Candidate summaries
    - Missing skills
    - ATS recommendations
    - Hiring verdicts

- 📈 Interactive Visualizations
  - Radar charts
  - Skill distribution graphs
  - Match analytics using Plotly & Matplotlib

- 📄 PDF Report Generation
  - Downloadable AI-generated analysis reports
  - Includes charts, recommendations, and insights

---

# 🏗️ System Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/bccb5818-5753-46f3-95df-241c9a608039" />

---

# 🧰 Tech Stack

- Python
- Streamlit
- Ollama
- Qwen2.5:3B
- FAISS
- all-MiniLM-L6-v2
- SentenceTransformers
- PyPDF2
- Plotly
- Matplotlib
- NumPy
- Pandas

---

# 📊 Core Functionalities

| Feature | Description |
|---|---|
| Semantic Matching | Context-aware resume-JD similarity |
| RAG Pipeline | Retrieves relevant chunks for analysis |
| Skill Gap Detection | Identifies missing technical skills |
| ATS Recommendations | Suggests resume improvements |
| Hiring Verdict | Strong Fit → Not a Fit |
| PDF Reports | Downloadable visual analysis |

---

# 📁 Project Structure

```bash
JobFit-AI/
│
├── app.py
├── core/
│   ├── pdf_reader.py
│   ├── embeddings.py
│   ├── faiss_store.py
│   ├── rag_pipeline.py
│   ├── llm_engine.py
│   ├── analytics.py
│   └── report_generator.py
│
├── outputs/
│   ├── reports/
│   └── charts/
│
├── requirements.txt
└── README.md
```

---



# 📌 Key Highlights

- Built a complete AI recruitment analysis pipeline
- Combined Semantic Search + RAG + Local LLMs
- Designed fully offline with zero API dependency
- Improved contextual understanding beyond keyword matching
- Generated structured JSON-based hiring insights
- Integrated interactive analytics and report generation

---

# 🔮 Future Improvements

- Multi-resume ranking system
- Domain-specific recruitment models
- Interview question generation
- Fine-tuned recruitment LLM
- Docker & cloud deployment
- Resume optimization assistant

---

# 👨‍💻 Author

Vedika Anand Thakur  
B.E. Information Technology  
Pune Institute of Computer Technology (PICT)

GitHub: https://github.com/your-username/jobfit-ai

---

# ⭐ If you like this project

Give this repository a star ⭐ and feel free to contribute!
