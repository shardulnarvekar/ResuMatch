import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import asyncio
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_openai_response(prompt, model="gpt-3.5-turbo", temperature=0.7, max_retries=2):
    """Enhanced OpenAI request handler with robust error handling and retry mechanism"""
    
    for retry_count in range(max_retries + 1):
        try:
            print(f"=== OPENAI REQUEST (Attempt {retry_count + 1}) ===")
            print(f"Model: {model}")
            print(f"Prompt length: {len(prompt)}")
            print(f"Temperature: {temperature}")
            
            # Add small delay between retries
            if retry_count > 0:
                await asyncio.sleep(2)
            
            # Enhanced system message for better response quality
            system_message = """You are an elite resume optimization expert with 15+ years of experience in recruitment, ATS systems, and career coaching. You have helped thousands of professionals land their dream jobs by creating compelling, keyword-optimized resumes.

Your expertise includes:
- Deep understanding of ATS (Applicant Tracking Systems) and how they scan resumes
- Knowledge of industry-specific keywords and requirements across all sectors
- Ability to identify gaps between candidate profiles and job requirements
- Strategic positioning of experience to maximize impact
- Quantification of achievements for maximum credibility
- Modern resume formatting and presentation best practices

Your responses are always:
- Highly specific and actionable
- Tailored to the exact job and candidate
- Focused on immediate, implementable changes
- Professional yet encouraging in tone
- Comprehensive but concise
- Results-oriented and strategic"""

            # Use the OpenAI client with proper async handling
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1800,
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
            )
            
            result = response.choices[0].message.content.strip()
            
            print("=== OPENAI RESPONSE RECEIVED ===")
            print(f"Response length: {len(result)}")
            print(f"Has strengths section: {'✨ STRENGTHS' in result}")
            print(f"Has improvements section: {'🔍 IMPROVEMENTS' in result}")
            print(f"Has pro tip section: {'💡 PRO TIP' in result}")
            print("Response preview:", result[:200] + "...")
            print("===============================")
            
            # Validate response quality
            if validate_response_quality(result):
                return result
            else:
                print(f"❌ Response quality check failed on attempt {retry_count + 1}")
                if retry_count < max_retries:
                    print("Retrying with modified parameters...")
                    # Adjust parameters for retry
                    temperature = min(0.9, temperature + 0.1)
                    continue
                else:
                    raise Exception("Response quality validation failed after all retries")

        except openai.RateLimitError as e:
            print(f"❌ Rate limit error on attempt {retry_count + 1}: {e}")
            if retry_count < max_retries:
                wait_time = (2 ** retry_count) * 2  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
                continue
            else:
                raise Exception(f"OpenAI rate limit exceeded after {max_retries + 1} attempts")

        except openai.APIError as e:
            print(f"❌ OpenAI API error on attempt {retry_count + 1}: {e}")
            if retry_count < max_retries:
                await asyncio.sleep(3)
                continue
            else:
                raise Exception(f"OpenAI API error after {max_retries + 1} attempts: {str(e)}")

        except Exception as e:
            print(f"❌ Unexpected error on attempt {retry_count + 1}: {e}")
            if retry_count < max_retries:
                await asyncio.sleep(2)
                continue
            else:
                raise Exception(f"OpenAI request failed after {max_retries + 1} attempts: {str(e)}")
    
    # This should never be reached, but just in case
    raise Exception("OpenAI request failed: Maximum retries exceeded")

def validate_response_quality(response):
    """Validate that the OpenAI response meets quality standards"""
    
    if not response or len(response.strip()) < 300:
        print("❌ Response too short")
        return False
    
    # Check for required sections
    required_sections = ['✨ STRENGTHS', '🔍 IMPROVEMENTS', '💡 PRO TIP']
    missing_sections = [section for section in required_sections if section not in response]
    
    if missing_sections:
        print(f"❌ Missing required sections: {missing_sections}")
        return False
    
    # Check for minimum content in each section
    sections = response.split('✨ STRENGTHS')[1] if '✨ STRENGTHS' in response else ""
    if '🔍 IMPROVEMENTS' in sections:
        strengths_section = sections.split('🔍 IMPROVEMENTS')[0]
        improvements_section = sections.split('🔍 IMPROVEMENTS')[1]
        
        if '💡 PRO TIP' in improvements_section:
            improvements_section = improvements_section.split('💡 PRO TIP')[0]
        
        # Validate strengths section
        if len(strengths_section.strip()) < 100:
            print("❌ Strengths section too short")
            return False
        
        # Validate improvements section
        if len(improvements_section.strip()) < 200:
            print("❌ Improvements section too short")
            return False
        
        # Count bullet points in improvements
        improvement_bullets = improvements_section.count('•')
        if improvement_bullets < 4:
            print(f"❌ Insufficient improvement suggestions: {improvement_bullets}")
            return False
    
    # Check for pro tip section
    if '💡 PRO TIP' in response:
        pro_tip_section = response.split('💡 PRO TIP')[1]
        if len(pro_tip_section.strip()) < 50:
            print("❌ Pro tip section too short")
            return False
    
    # Check for generic/template responses
    generic_phrases = [
        "customize your resume",
        "tailor your application", 
        "add more keywords",
        "improve your resume"
    ]
    
    generic_count = sum(1 for phrase in generic_phrases if phrase.lower() in response.lower())
    if generic_count > 2:
        print(f"❌ Response appears too generic (generic phrases: {generic_count})")
        return False
    
    print("✅ Response quality validation passed")
    return True

async def test_openai_connection():
    """Test OpenAI connection and API key validity"""
    try:
        print("=== TESTING OPENAI CONNECTION ===")
        
        test_prompt = "Respond with exactly: 'OpenAI connection successful'"
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=50
            )
        )
        
        result = response.choices[0].message.content.strip()
        
        if "OpenAI connection successful" in result:
            print("✅ OpenAI connection test passed")
            return True
        else:
            print(f"❌ OpenAI connection test failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI connection test failed: {e}")
        return False

async def get_enhanced_suggestions(job_description, resume_text, similarity_score, matched_keywords, missing_keywords):
    """Generate enhanced suggestions with multiple fallback strategies"""
    
    primary_prompt = f"""As a senior resume consultant, analyze this resume against the job posting and provide detailed optimization advice.

JOB POSTING:
{job_description[:2000]}

CANDIDATE'S RESUME:
{resume_text[:2000]}

CURRENT ANALYSIS:
- Match Score: {similarity_score:.1f}%
- Matched Keywords: {', '.join(matched_keywords[:10])}
- Missing Keywords: {', '.join(missing_keywords[:8])}

Please provide comprehensive optimization recommendations in this exact format:

✨ STRENGTHS
• [Identify specific strengths from their resume that align with this job]
• [Highlight relevant achievements or experiences that make them competitive]
• [Note any unique qualifications or standout elements]
• [Mention skills or background that directly match job requirements]

🔍 IMPROVEMENTS NEEDED
• Keyword Integration: [Specific guidance on incorporating missing keywords naturally]
• Quantification: [Suggest specific metrics and numbers they should add to achievements]
• Experience Positioning: [How to better showcase their most relevant experience]
• Skills Enhancement: [Technical or soft skills to emphasize or add]
• Resume Structure: [Formatting, organization, or presentation improvements]
• Language Optimization: [More impactful words, action verbs, or industry terminology]
• Content Additions: [Certifications, projects, or experiences to highlight]
• Tailoring Strategy: [How to customize this resume specifically for this role]

💡 PRO TIP
[Provide strategic, actionable advice specific to this candidate and role. Include exact keywords to incorporate and tactical tips for maximizing their chances with this specific job posting. Focus on the 1-2 most impactful changes they can make.]

Make every recommendation specific, actionable, and tailored to this exact job and candidate combination."""

    try:
        # First attempt with primary prompt
        suggestions = await get_openai_response(primary_prompt, temperature=0.7)
        
        if suggestions and len(suggestions.strip()) >= 400:
            return suggestions
        
        # If first attempt insufficient, try with different parameters
        print("First attempt insufficient, trying enhanced approach...")
        
        enhanced_prompt = f"""You are an expert resume optimization consultant. This candidate needs specific help improving their resume for this job.

JOB: {job_description[:1500]}
RESUME: {resume_text[:1500]}
SCORE: {similarity_score:.1f}%
MATCHED: {', '.join(matched_keywords[:8])}
MISSING: {', '.join(missing_keywords[:6])}

Provide detailed recommendations:

✨ STRENGTHS
• [List 4 specific strengths from their resume relevant to this job]

🔍 IMPROVEMENTS NEEDED
• [Provide 8 specific, actionable improvements they can make]

💡 PRO TIP
[Give strategic advice specific to this role and candidate]

Be specific, actionable, and focus on what will most improve their chances for this exact position."""

        suggestions = await get_openai_response(enhanced_prompt, temperature=0.8)
        
        if suggestions and len(suggestions.strip()) >= 300:
            return suggestions
        
        # Final fallback attempt
        print("Second attempt insufficient, trying final approach...")
        
        simple_prompt = f"""Analyze this resume for the job and provide optimization advice:

JOB: {job_description[:1000]}
RESUME: {resume_text[:1000]}

Format your response as:
✨ STRENGTHS
• [3-4 strengths]

🔍 IMPROVEMENTS NEEDED  
• [6-8 improvements]

💡 PRO TIP
[Strategic advice]"""

        return await get_openai_response(simple_prompt, temperature=0.9)
        
    except Exception as e:
        print(f"All OpenAI attempts failed: {e}")
        raise Exception("Failed to generate AI suggestions after multiple strategic attempts")