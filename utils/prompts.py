# project/utils/prompts.py
class Prompts:
    @staticmethod
    def extract_skills_prompt(text):
        return f"""
Extract a list of key skills from the following text as JSON: {{"skills": ["skill1", "skill2", ...]}}

Text: {text}
"""

    @staticmethod
    def analysis_prompt(context, jd, jd_skills, resume_skills):
        return f"""
Analyze the resume context against the job description and provided skills.
Output JSON with:
{{
  "match_score": integer 0-100,
  "matching_skills": array of strings,
  "missing_skills": array of strings,
  "summary": string,
  "recommendations": string (include ATS improvements),
  "verdict": string (e.g., "Shortlist", "Reject", "Maybe")
}}

JD Skills: {', '.join(jd_skills)}
Resume Skills: {', '.join(resume_skills)}
Job Description: {jd}
Resume Context: {context}
"""
