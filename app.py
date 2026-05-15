# app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import streamlit as st

from core.pdf_reader import PDFReader
from core.rag_pipeline import RAGPipeline
from core.llm_engine import LLMEngine
from core.analytics import Analytics
from core.report_generator import ReportGenerator
from utils.text_cleaning import TextCleaner

# === Page Config & Styling ===
st.set_page_config(page_title="JobFit AI — GenAI Resume Matcher", layout="wide")

# Load custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# === Title & Free Mode Banner ===
st.title("JobFit AI — GenAI Resume Matcher")
st.success("JobFit AI is an AI powered resume analysis tool that evaluates how well your resume matches a job description. Upload your resume and paste a job description to get a detailed analysis, including a match score, skill comparison, and personalized recommendations for improvement. Perfect for job seekers looking to optimize their resumes for ATS and recruiters seeking efficient candidate screening.")

# === UI Layout ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Resume (PDF)")
    uploaded_file = st.file_uploader("Drag and drop or browse", type="pdf", label_visibility="collapsed")

with col2:
    st.subheader("Job Description")
    jd = st.text_area("Paste the job description here", height=200, 
                     placeholder="e.g. We are looking for a Python developer with experience in FastAPI, Docker...")

# === Main Analysis Button ===
if st.button("Analyze Resume", type="primary", use_container_width=True):
    if not uploaded_file:
        st.error("Please upload a PDF resume.")
    elif not jd.strip():
        st.error("Please provide a job description.")
    else:
        # --- PDF Reading ---
        with st.spinner("Extracting text from resume..."):
            try:
                pdf_text = PDFReader.extract_text(uploaded_file)
                clean_text = TextCleaner.clean(pdf_text)
                if len(clean_text.strip()) < 100:
                    st.error("Resume text is too short or unreadable. Try another PDF.")
                    st.stop()
            except Exception as e:
                st.error(f"Error reading PDF: {str(e)}")
                st.stop()

        # --- RAG Pipeline ---
        with st.spinner("Chunking resume & building local vector database..."):
            try:
                rag = RAGPipeline()
                chunks = rag.chunk_text(clean_text)
                if not chunks:
                    st.error("No content found in resume.")
                    st.stop()
                
                # Batch encoding (much faster)
                embeddings = rag.embedder.model.encode(
                    chunks, 
                    convert_to_numpy=True, 
                    normalize_embeddings=True,
                    batch_size=32
                )
                embeddings = np.array(embeddings)  # Ensure numpy array
                
                rag.build_index(embeddings, chunks)
                
            except Exception as e:
                st.error(f"RAG setup failed: {str(e)}")
                st.stop()

        # --- LLM Analysis ---
        with st.spinner("Analyzing with local AI model (this may take 10–30 seconds)..."):
            analysis = None
            analysis_str = None
            
            try:
                llm = LLMEngine()
                query_emb = rag.embedder.embed(jd)
                retrieved_chunks = rag.retrieve(query_emb, k=6)
                context = "\n\n".join(retrieved_chunks)

                # Extract skills
                jd_skills = llm.extract_skills(jd)
                resume_skills = llm.extract_skills(clean_text)

                # Final analysis
                analysis_str = llm.analyze(context, jd, jd_skills, resume_skills)
                analysis = json.loads(analysis_str)

            except json.JSONDecodeError as e:
                error_text = analysis_str[:700] if analysis_str else "No output"
                st.error(f"AI returned invalid JSON. Raw output:\n\n{error_text}")
                st.stop()
            except Exception as e:
                st.error(f"LLM analysis failed: {str(e)}")
                if analysis_str:
                    with st.expander("Raw AI Output"):
                        st.code(analysis_str)
                st.stop()

        # === SUCCESS! Display Results ===
        if analysis is None:
            st.error("Analysis could not be completed.")
            st.stop()

        st.success("Analysis Complete!")

        # Score & Verdict
        col_score, col_verdict = st.columns(2)
        with col_score:
            st.markdown(f"<div class='highlight-box'><h2>{analysis.get('match_score', 0)}/100</h2><p>Match Score</p></div>", unsafe_allow_html=True)
        
        with col_verdict:
            verdict = analysis.get('verdict', 'Unknown')
            color = "green" if "Fit" in verdict else "orange" if "Moderate" in verdict else "red"
            st.markdown(f"<div class='highlight-box' style='border-color:{color}'><h3>{verdict}</h3><p>Shortlisting Verdict</p></div>", unsafe_allow_html=True)

        # Skills
        col_match, col_miss = st.columns(2)

        with col_match:
            st.success(f"Matching Skills ({len(analysis.get('matching_skills', []))})")
            matching = analysis.get("matching_skills", [])
            
            if matching:
                st.markdown("\n".join([f"- {skill}" for skill in matching]))
            else:
                st.write("None detected")

        with col_miss:
            st.warning(f"Missing Skills ({len(analysis.get('missing_skills', []))})")
            missing = analysis.get("missing_skills", [])
            
            if missing:
                st.markdown("\n".join([f"- {skill}" for skill in missing]))
            else:
                st.write("None required")
        # Summary & Recommendations
        with st.expander("Summary", expanded=True):
            st.write(analysis.get('summary', 'No summary provided.'))

        # with st.expander("Recommendations & ATS Tips", expanded=True):
        #     st.write(analysis.get('recommendations', 'No recommendations provided.'))
        with st.expander("💡 Recommendations & ATS Tips", expanded=True):
            st.markdown(analysis.get('recommendations', 'No recommendations available.'))

        # with st.expander("Raw AI Output (for debugging)"):
        #     st.json(analysis)

        # # === Visual Insights ===
                # === PROFESSIONAL VISUAL INSIGHTS ===
        st.subheader("📊 Match Analysis Results")
        
        analytics = Analytics()
        
        # Match Score Gauge
        st.plotly_chart(analytics.match_score_gauge(analysis.get('match_score', 0)), 
                       use_container_width=True)

        # Radar + Skills Bar
        col1, col2 = st.columns(2)
        with col1:
            # Example Radar (you can make this dynamic later)
            categories = ['Skills', 'Projects', 'Experience', 'Education']
            values = [8, 7, 6, 5]   # You can improve this later from LLM
            st.plotly_chart(analytics.radar_chart(categories, values), use_container_width=True)
        
        with col2:
            st.plotly_chart(analytics.skill_comparison(
                analysis.get('matching_skills', []), 
                analysis.get('missing_skills', [])
            ), use_container_width=True)

        
        # === Download Report ===
        st.subheader("Download Report")
        report_gen = ReportGenerator()
        report_pdf = report_gen.generate_report(analysis, jd, clean_text)
        st.download_button(
            "Download Full Report (PDF)",
            report_pdf,
            file_name=f"JobFit_Report_{analysis.get('match_score', 0)}_Score.pdf",
            mime="application/pdf"
        )

        st.balloons()
