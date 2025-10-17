from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Load environment variables
load_dotenv()
genai.configure(api_key="AIzaSyDspZYjboZwXGo7VLnTPW0Q_miuDFzbhro")

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
            4. **Recommendation**: How can the candidate improve the resume to increase job match?
            5. **Verdict**: Would you shortlist this candidate?

            Format your response with clear headings and bullet points.
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
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=["Match Score"], y=[match_score], palette="Blues_d", ax=ax)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Match Percentage")
        ax.set_title("Resume-Job Match Score")
        ax.bar_label(ax.containers[0], fontsize=12, color="black", padding=5)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_str

    @staticmethod
    def generate_skills_chart(matching_skills, missing_skills):
        skills_data = {"Matching Skills": len(matching_skills), "Missing Skills": len(missing_skills)}
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=list(skills_data.keys()), y=list(skills_data.values()), palette="Set2", ax=ax)
        ax.set_ylabel("Number of Skills")
        ax.set_title("Skills Overview")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_str

    @staticmethod
    def generate_skills_pie_chart(matching_skills, missing_skills):
        labels = ['Matching Skills', 'Missing Skills']
        sizes = [len(matching_skills), len(missing_skills)]
        colors = ['#66bb6a', '#ef5350']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        ax.set_title('Skill Match Ratio')
        ax.axis('equal')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_str

# ---------------------------
# Streamlit App Layout
# ---------------------------
def main():
    st.set_page_config(page_title="Resume Job Matcher", layout="wide", page_icon="🧠")

    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2838/2838912.png", width=100)
        st.title("Resume Matcher")
        st.markdown("🔍 Upload your resume and match it with any job description.")
        st.markdown("📈 Get a smart analysis with match score, feedback & tips.")
        st.markdown("💡 Powered by **Gemini AI**")
        st.markdown("---")
        st.info("Tip: Use a well-formatted resume for best results.")

    # Main Title
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 Resume Job Matcher</h1>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: gray;'>Get personalized match insights and recommendations</h5>", unsafe_allow_html=True)

    # Layout
    st.markdown("## 📥 Upload & Describe")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"], help="Upload your latest resume in PDF format.")

    with col2:
        job_description = st.text_area("Paste Job Description", height=250, help="Copy the full job description from any job portal or email.")

    # Process & Output
    if uploaded_file and job_description:
        st.success("✅ Resume and job description received.")

        if st.button("🚀 Match Now"):
            with st.spinner("Analyzing resume and job description..."):
                pdf_text = ResumeJobMatcher.extract_text_from_pdf(uploaded_file)

                if pdf_text:
                    result = ResumeJobMatcher.get_match_analysis(pdf_text, job_description)

                    # Placeholder values (you can extract these from the result in future improvements)
                    match_score = 85
                    matching_skills = ["Python", "SQL", "Machine Learning"]
                    missing_skills = ["Java", "AWS"]

                    if result:
                        st.markdown("## 🎯 Match Report")
                        st.markdown("<div style='background-color: #2e4053; padding: 20px; border-radius: 10px;'>"
                                    f"<pre style='white-space: pre-wrap; font-size: 16px'>{result}</pre>"
                                    "</div>", unsafe_allow_html=True)

                        # Match Score Chart
                        match_score_img = ResumeJobMatcher.generate_match_score_chart(match_score)
                        st.markdown(f"### Match Score Visualization")
                        st.image(f"data:image/png;base64,{match_score_img}", caption="Match Score")

                        # Skills Overview
                        st.markdown("### Skills Overview")
                        col_chart1, col_chart2 = st.columns(2)

                        with col_chart1:
                            bar_img = ResumeJobMatcher.generate_skills_chart(matching_skills, missing_skills)
                            st.image(f"data:image/png;base64,{bar_img}", caption="Bar Chart of Skills")

                        with col_chart2:
                            pie_img = ResumeJobMatcher.generate_skills_pie_chart(matching_skills, missing_skills)
                            st.image(f"data:image/png;base64,{pie_img}", caption="Pie Chart of Skill Match")

                        # Download Report
                        st.download_button(
                            label="📥 Download Match Report",
                            data=result,
                            file_name="match_report.txt",
                            mime="text/plain",
                        )
                else:
                    st.warning("Could not extract text from the PDF.")
    else:
        st.info("Upload a resume and provide a job description to begin.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center;'>Crafted with ❤️ using Streamlit & Gemini AI | <a href='https://cloud.google.com/ai'>Google AI</a></p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
