import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_resume_improvement_suggestions(resume_text: str, job_description: str) -> list:
    """
    Get AI-powered suggestions for improving a resume based on a job description
    """
    try:
        prompt = f"""
        As an AI career coach, analyze this resume and job description. 
        Provide 3-5 specific, actionable suggestions to improve the resume to better match the job.
        Focus on missing skills, experience, or keywords. Be specific and concise.
        
        RESUME:
        {resume_text}
        
        JOB DESCRIPTION:
        {job_description}
        
        SUGGESTIONS:
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        suggestions_text = response.choices[0].message.content.strip()
        
        # Parse the suggestions into a list
        suggestions = []
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or 
                        any(line.startswith(f"{i}.") for i in range(1, 10))):
                # Remove the bullet point or number and strip
                suggestions.append(line[line.find(' ')+1:].strip())
        
        # If parsing failed, return the whole text as one suggestion
        if not suggestions:
            suggestions = [suggestions_text]
            
        return suggestions
            
    except Exception as e:
        print(f"Error getting OpenAI suggestions: {str(e)}")
        return ["Unable to generate AI suggestions at this time."]


def get_openai_response(prompt, model="gpt-3.5-turbo"):
    """General function to get a response from OpenAI"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()