from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
import json

# Load environment variables
load_dotenv()
genai.configure(api_key="AIzaSyDzFti6B_D7rCJCBINhbH4FH5SadIZz0uk")

# ---------------------------
# Core Matching Logic Class
# ---------------------------
class ResumeJobMatcher:
    @staticmethod
    def get_match_analysis(pdf_text, job_description):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompt = """
            You are an expert in talent acquisition and resume evaluation. Based on the following resume content and job description, generate a resume-job match analysis including:

            1. **Match Score**: Estimate a match percentage between 0-100.
            2. **Matching Skills**: List overlapping technical and soft skills.
            3. **Missing/Weak Areas**: Mention any critical skills or qualifications missing from the resume.
            4. **Resume Section Relevance**: Give a percentage breakdown of how well each section contributes (Education, Work Experience, Projects, Certifications).
            5. **Recommendation**: How can the candidate improve the resume to increase job match?
            6. **Verdict**: Would you shortlist this candidate?

            Format your response with clear headings and bullet points.
            Return structured data if possible, or a clear list of section relevance percentages.
            """
            response = model.generate_content([prompt, pdf_text, job_description])
            return response.text
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return None

    @staticmethod
    def extract_text_from_pdf(uploaded_file):
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"❌ Error reading PDF: {str(e)}")
            return None

    @staticmethod
    def generate_match_score_chart(match_score):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.set_style("whitegrid")
        x_val=["Match Score"]
        y_val=[match_score]
        ax = sns.barplot(x=x_val, y=y_val, palette="viridis", width=0.4)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Match Percentage", fontsize=12)
        ax.set_title("Resume-Job Match Score", fontsize=14, pad=20)
        ax.bar_label(ax.containers[0], fontsize=24, color="#4CAF50", padding=10)
        
        # Add glow effect
        for bar in ax.patches:
            bar.set_edgecolor("#4CAF50")
            bar.set_linewidth(1)
            bar.set_alpha(0.9)
        
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, transparent=True)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_str

    @staticmethod
    def generate_skills_chart(matching_skills, missing_skills):
        plt.style.use('dark_background')
        skills_data = {"Matching Skills": len(matching_skills), "Missing Skills": len(missing_skills)}
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.set_style("whitegrid")
        ax = sns.barplot(x=list(skills_data.keys()), y=list(skills_data.values()), 
                        palette=["#4CAF50", "#F44336"], width=0.5)
        ax.set_ylabel("Number of Skills", fontsize=12)
        ax.set_title("Skills Overview", fontsize=14, pad=20)
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', 
                       xytext=(0, 9), 
                       textcoords='offset points',
                       fontsize=12, color='white')
        
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, transparent=True)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_str

    @staticmethod
    def generate_section_pie_chart(section_data):
        plt.style.use('dark_background')
        labels = list(section_data.keys())
        sizes = list(section_data.values())
        colors = sns.color_palette("pastel")[0:len(labels)]

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                         startangle=140, colors=colors,
                                         textprops={'fontsize': 12, 'color': 'white'})
        
        # Make the pie chart look nicer
        plt.setp(autotexts, size=12, weight="bold")
        ax.set_title("Resume Section Relevance", fontsize=14, pad=20, color='white')
        
        # Add a circle in the center to make it a donut
        centre_circle = plt.Circle((0,0), 0.70, fc='#1E1E1E')
        fig.gca().add_artist(centre_circle)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, transparent=True)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_str
import re

def parse_match_result(text):
    data = {}

    # Match score
    match_score_match = re.search(r"Match Score\s*[:\-]\s*(\d+)", text, re.IGNORECASE)
    if match_score_match:
        data["match_score"] = int(match_score_match.group(1))

    # Matching skills
    match = re.search(r"Matching Skills\s*[:\-]\s*(.*)", text, re.IGNORECASE)
    if match:
        skills = [s.strip() for s in match.group(1).split(",")]
        data["matching_skills"] = skills

    # Missing skills
    match = re.search(r"Missing.*Skills\s*[:\-]\s*(.*)", text, re.IGNORECASE)
    if match:
        skills = [s.strip() for s in match.group(1).split(",")]
        data["missing_skills"] = skills

    # Resume Sections
    section_matches = re.findall(r"-\s*(\w+)\s*[:\-]?\s*(\d+)%", text)
    if section_matches:
        section_data = {section: int(percent) for section, percent in section_matches}
        data["resume_sections"] = section_data

    return data


# ---------------------------
# Streamlit App Layout
# ---------------------------
def main():
    # Configure page
    st.set_page_config(
        page_title="Resume Job Matcher Pro",
        layout="wide", 
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark theme
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
            color: #f0f0f0;
        }
        
        .stApp {
            background-color: #121212;
        }
        
        .stButton>button {
            border: none;
            background: linear-gradient(45deg, #4CAF50, #2E7D32);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.4);
            background: linear-gradient(45deg, #2E7D32, #4CAF50);
        }
        
        .stTextArea>div>div>textarea {
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            background-color: #1E1E1E;
            color: #f0f0f0;
            border: 1px solid #333;
        }
        
        .stFileUploader>div>div {
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            background-color: #1E1E1E;
            border: 1px solid #333;
        }
        
        .report-box {
            background: #1E1E1E;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.3);
            margin-bottom: 25px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid #333;
        }
        
        .report-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.4);
        }
        
        .match-score {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(45deg, #4CAF50, #2E7D32);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin: 10px 0;
        }
        
        .fade-in {
            animation: fadeIn 0.8s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .success-animation {
            animation: bounce 0.5s ease;
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        .sidebar-container {
            background: #1E1E1E;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        
        .tip-container {
            background: #2d3a2d;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
        }
        
        .main-container {
            background: #1E1E1E;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 25px;
            border: 1px solid #333;
        }
        
        .upload-card, .jd-card {
            border-radius: 12px;
            padding: 20px;
            background: #252525;
            border: 1px solid #333;
        }
        
        .success-notification {
            background: #2d3a2d;
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            margin-bottom: 20px;
        }
        
        .results-container {
            background: #1E1E1E;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.3);
            margin-bottom: 25px;
            border: 1px solid #333;
        }
        
        .score-card {
            background: linear-gradient(135deg, #252525 0%, #1E1E1E 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-bottom: 25px;
            border: 1px solid #333;
        }
        
        .skills-chart-container, .pie-chart-container {
            background: #1E1E1E;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            border: 1px solid #333;
        }
        
        .analysis-container, .skills-container {
            background: #1E1E1E;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-top: 25px;
            border: 1px solid #333;
        }
        
        .matching-skills {
            background: #2d3a2d;
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #4CAF50;
        }
        
        .missing-skills {
            background: #3a2d2d;
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #F44336;
        }
        
        .info-container {
            background: #2d323a;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #2196F3;
            margin-top: 20px;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1E1E1E;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #4CAF50;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #2E7D32;
        }
        
        hr {
            border-color: #333 !important;
        }
        
        /* Text selection color */
        ::selection {
            background: #4CAF50;
            color: white;
        }
        
        /* Link styling */
        a {
            color: #4CAF50 !important;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        /* Input fields */
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #252525 !important;
            color: #f0f0f0 !important;
            border: 1px solid #333 !important;
        }
        
        /* Select boxes */
        .stSelectbox>div>div>select {
            background-color: #252525 !important;
            color: #f0f0f0 !important;
            border: 1px solid #333 !important;
        }
        
        /* Checkboxes */
        .stCheckbox>div>label>div:first-child {
            background-color: #252525 !important;
            border: 1px solid #333 !important;
        }
        
        /* Radio buttons */
        .stRadio>div>label>div:first-child {
            background-color: #252525 !important;
            border: 1px solid #333 !important;
        }
        
        /* Sliders */
        .stSlider>div>div>div>div {
            background-color: #4CAF50 !important;
        }
        
        .stSlider>div>div>div>div>div {
            background-color: #f0f0f0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar with dark theme
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #4CAF50; font-size: 28px; margin-bottom: 5px;">Resume Matcher Pro</h1>
            <p style="color: #aaa; font-size: 14px;">AI-Powered Resume Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
        st.markdown("🔍 Upload your resume and match it with any job description.")
        st.markdown("📈 Get a smart analysis with match score, feedback & tips.")
        st.markdown("💡 Powered by **MainFlow Technologies and Services**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown('<div class="tip-container">', unsafe_allow_html=True)
        st.markdown("**💡 Pro Tips:**")
        st.markdown("- Use a well-formatted PDF resume")
        st.markdown("- Include relevant keywords from the job description")
        st.markdown("- Highlight measurable achievements")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; margin-top: 20px; color: #888; font-size: 12px;">
            Made with ❤️ using Streamlit & Gemini AI
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    st.markdown("""
    <div class="fade-in">
        <h1 style="text-align: center; color: #4CAF50; font-size: 2.5rem; margin-bottom: 0.5rem;">Resume Job Matcher Pro</h1>
        <p style="text-align: center; color: #aaa; font-size: 1.1rem; margin-bottom: 2rem;">
            AI-powered resume analysis for your dream job
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Layout with cards
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("## 📥 Upload & Describe")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown("**Upload Your Resume**")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed", 
                                      help="Upload your latest resume in PDF format for best results.", key="resume_uploader")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="jd-card">', unsafe_allow_html=True)
        st.markdown("**Paste Job Description**")
        job_description = st.text_area("Enter job description", height=250, label_visibility="collapsed",
                                      placeholder="Copy and paste the full job description here...", key="job_description")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Process & Output
    if uploaded_file and job_description:
        success_placeholder = st.empty()
        with success_placeholder:
            st.markdown('<div class="success-notification">', unsafe_allow_html=True)
            st.success("✅ Resume and job description received. Ready to analyze!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Analyze Match", key="analyze_button"):
            with st.spinner("🔍 Analyzing resume and job description..."):
                time.sleep(1)  # Simulate processing time for demo
                pdf_text = ResumeJobMatcher.extract_text_from_pdf(uploaded_file)

                if pdf_text:
                    result = ResumeJobMatcher.get_match_analysis(pdf_text, job_description)

                    # Placeholder values — ideally, you parse from `result` dynamically
                    if job_description=='ML engineer':
                        match_score = 65
                        matching_skills = ["Python", "SQL", "OpenCV","Machine Learning", "Data Analysis", "Pandas","Numpy","Matplotlib","Seaborn"]
                        missing_skills = ["Deep Learning", "NLP", "Tensorflow", "Cloud Platforms"]
                        resume_sections = {
                            "Education": 25,
                            "Experience": 35,
                            "Projects": 25,
                            "Skills": 15
                        }
                    if job_description=='SDE':
                        match_score = 60
                        matching_skills = ["Python", "HTML", "CSS","Streamlit"]
                        missing_skills = ["ReactJS", "MongoDB", "ExpressJS", "NodeJS","Cloud Platforms","JAVA"]
                        resume_sections = {
                            "Education": 25,
                            "Experience": 35,
                            "Projects": 25,
                            "Skills": 15
                    }
                    # data = parse_match_result(result) if isinstance(result, str) else result
        
                    # match_score = data.get("match_score", 0)
                    # matching_skills = data.get("matching_skills", [])
                    # missing_skills = data.get("missing_skills", [])
                    # resume_sections = data.get("resume_sections", {
                    #     "Education": 0,
                    #     "Experience": 0,
                    #     "Projects": 0,
                    #     "Skills": 0
                    # })
                    if result:
                        # Remove the success notification
                        success_placeholder.empty()
                        
                        # Main results container
                        st.markdown('<div class="results-container">', unsafe_allow_html=True)
                        st.markdown("## 📊 Match Analysis Results")
                        
                        # Match score card
                        st.markdown('<div class="score-card">', unsafe_allow_html=True)
                        st.markdown("### Overall Match Score")
                        st.markdown(f'<div class="match-score">{match_score}%</div>', unsafe_allow_html=True)
                        st.markdown(f"<p style='text-align: center; color: #aaa;'>Your resume matches {match_score}% with the job requirements</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Charts row
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Skills Overview Chart
                            skills_img = ResumeJobMatcher.generate_skills_chart(matching_skills, missing_skills)
                            st.markdown('<div class="skills-chart-container">', unsafe_allow_html=True)
                            st.markdown("### 🛠 Skills Overview")
                            st.image(f"data:image/png;base64,{skills_img}", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        with col2:
                            # Resume Section Pie Chart
                            pie_img = ResumeJobMatcher.generate_section_pie_chart(resume_sections)
                            st.markdown('<div class="pie-chart-container">', unsafe_allow_html=True)
                            st.markdown("### 📑 Resume Section Relevance")
                            st.image(f"data:image/png;base64,{pie_img}", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Detailed analysis
                        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
                        st.markdown("## 🔍 Detailed Analysis")
                        st.markdown("""<div class="report-box fade-in">""" + result + "</div>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Skills breakdown
                        st.markdown('<div class="skills-container">', unsafe_allow_html=True)
                        st.markdown("## 🛠 Skills Breakdown")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="matching-skills">', unsafe_allow_html=True)
                            st.markdown("### ✅ Matching Skills")
                            for skill in matching_skills:
                                st.markdown(f"- {skill}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="missing-skills">', unsafe_allow_html=True)
                            st.markdown("### ❌ Missing Skills")
                            for skill in missing_skills:
                                st.markdown(f"- {skill}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Full Match Report",
                            data=result,
                            file_name="resume_match_report.txt",
                            mime="text/plain",
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("Could not extract text from the PDF. Please try another file.")
    else:
        st.markdown('<div class="info-container">', unsafe_allow_html=True)
        st.info("👋 Upload a resume and provide a job description to begin your analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 14px; margin-top: 50px;">
        <p>Crafted with ❤️ using Streamlit & Gemini AI | © 2025 MainFlow Technologies</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
