
# core/llm_engine.py
import ollama
import json

class LLMEngine:
    def __init__(self):
        self.model = 'qwen2.5:3b'   # Best balance of speed & quality

    def _generate(self, prompt):
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']

    def extract_skills(self, text):
        prompt = f"Extract only key technical and soft skills as JSON list.\nText: {text[:5000]}\nOutput only: {{\"skills\": [\"skill1\", \"skill2\"]}}"
        try:
            resp = self._generate(prompt)
            start = resp.find('{')
            end = resp.rfind('}') + 1
            if start != -1:
                data = json.loads(resp[start:end])
                return data.get('skills', [])
            return []
        except:
            return []

    def analyze(self, context, jd, jd_skills, resume_skills):
        prompt = f"""
You are a senior technical recruiter. Do a **detailed** analysis.

Job Description:
{jd}

Resume Context:
{context}

JD Skills: {', '.join(jd_skills) if jd_skills else 'None'}
Resume Skills: {', '.join(resume_skills) if resume_skills else 'None'}

Return **ONLY** valid JSON:

{{
  "match_score": <0-100>,
  "matching_skills": [<list>],
  "missing_skills": [<list>],
  "verdict": "Strong Fit" | "Good Fit" | "Moderate Fit" | "Not a Fit",
  "summary": "Write a detailed 15-20 sentence summary covering:
   - Overall fit
   - Relevance and strength of Projects
   - Experience (relevance, strength, years if mentioned)
   - Education relevance and strength
   - Key strengths and critical gaps",
  "recommendations": "3-5 specific recommendations with ATS tips"
}}
"""
        response = self._generate(prompt)
        
        start = response.find('{')
        end = response.rfind('}') + 1
        json_str = response[start:end] if start != -1 else response
        return json_str
