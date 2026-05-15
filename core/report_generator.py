# core/report_generator.py
from fpdf import FPDF

class ReportGenerator:
    def generate_report(self, analysis, jd, resume_text):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        pdf.cell(200, 10, txt="JobFit AI Report", ln=1, align='C')
        pdf.ln(10)

        # Match Score & Verdict
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Match Score:", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=str(analysis.get('match_score', 0)), ln=1)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Verdict:", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=str(analysis.get('verdict', 'N/A')), ln=1)

        pdf.ln(10)
        
        # Skills
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Matching Skills:", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, ", ".join(analysis.get('matching_skills', [])))
        
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Missing Skills:", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, ", ".join(analysis.get('missing_skills', [])))

        pdf.ln(10)
        
        # Summary (Safe handling)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Detailed Summary:", ln=1)
        pdf.set_font("Arial", size=12)
        summary = analysis.get('summary', 'No summary available.')
        if isinstance(summary, list):
            summary = "\n".join(summary)
        pdf.multi_cell(0, 10, summary)

        pdf.ln(10)
        
        # Recommendations (Safe handling)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Recommendations & ATS Tips:", ln=1)
        pdf.set_font("Arial", size=12)
        recommendations = analysis.get('recommendations', 'No recommendations available.')
        if isinstance(recommendations, list):
            recommendations = "\n".join(recommendations)
        pdf.multi_cell(0, 10, recommendations)

        # Job Description & Resume
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Job Description:", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, jd)

        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Resume Text:", ln=1)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, resume_text[:10000])

        return pdf.output(dest='S').encode('latin-1')
