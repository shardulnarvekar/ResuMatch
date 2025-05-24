from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.gemini_client import get_gemini_response


# Initialize semantic model for deep text understanding
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Stopwords and punctuation
STOPWORDS = set(stopwords.words('english'))
PUNCTUATION = set(string.punctuation)

def preprocess_text(text):
    """Enhanced text preprocessing for better semantic understanding"""
    text = text.lower()
    # Preserve important technical terms and compound words
    text = re.sub(r'[^\w\s\-\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def extract_dynamic_keywords(job_description, context="job"):
    """Use Gemini to dynamically extract the most critical keywords from job description"""
    try:
        keyword_extraction_prompt = f"""You are an expert ATS (Applicant Tracking System) specialist and recruitment consultant. Your task is to extract ALL the critical and essential keywords from this job posting that would be used by ATS systems and hiring managers to filter candidates.

JOB POSTING:
{job_description}

INSTRUCTIONS:
1. Extract 15-20 of the MOST CRITICAL keywords/phrases including:
   - ALL technical skills, tools, software, programming languages, frameworks mentioned
   - ALL specific applications, platforms, systems named (like Salesforce, SAP, Jira, etc.)
   - Core required qualifications and certifications
   - Must-have experience areas and methodologies
   - Industry-specific terminology and acronyms
   - Specific degree requirements or experience levels

2. Be COMPREHENSIVE - include every specific tool, application, or technology mentioned, even if mentioned only once

3. Include both single words and multi-word phrases (like "machine learning", "project management")

4. Pay special attention to:
   - Brand names and proprietary software
   - Programming languages and frameworks
   - Cloud platforms and services
   - Methodologies and processes
   - Certifications and qualifications

5. Extract the EXACT terms as they appear in the job posting (maintain proper capitalization for brand names)

6. If a skill or tool is mentioned anywhere in the job description, include it in your list

FORMAT YOUR RESPONSE AS A SIMPLE COMMA-SEPARATED LIST:
keyword1, keyword2, keyword3, etc.

EXAMPLE OUTPUT:
Python, React, AWS, Salesforce, machine learning, project management, SQL, Docker, Agile methodology, Bachelor's degree, 3+ years experience, API development, data analysis, Kubernetes, Jenkins

YOUR EXTRACTED KEYWORDS:"""

        response = get_gemini_response(keyword_extraction_prompt)
        
        if response and response.strip():
            # Parse the comma-separated keywords
            keywords = [kw.strip().lower() for kw in response.split(',') if kw.strip()]
            # Remove duplicates while preserving order and limit to top 15
            seen = set()
            unique_keywords = []
            for kw in keywords:
                if kw not in seen and len(kw) > 1:
                    seen.add(kw)
                    unique_keywords.append(kw)
            
            return unique_keywords[:20]
        else:
            print("Failed to extract keywords via Gemini, using fallback")
            return extract_fallback_keywords(job_description)
            
    except Exception as e:
        print(f"Error in dynamic keyword extraction: {e}")
        return extract_fallback_keywords(job_description)

def extract_fallback_keywords(job_description):
    """Fallback keyword extraction using text analysis"""
    text = preprocess_text(job_description)
    words = text.split()
    
    # Look for common technical and professional terms
    important_patterns = [
        r'\b\w*python\w*\b', r'\b\w*java\w*\b', r'\b\w*react\w*\b', r'\b\w*sql\w*\b',
        r'\b\w*aws\w*\b', r'\b\w*cloud\w*\b', r'\b\w*api\w*\b', r'\b\w*database\w*\b',
        r'\b\w*management\w*\b', r'\b\w*analysis\w*\b', r'\b\w*development\w*\b',
        r'\b\w*experience\w*\b', r'\b\w*degree\w*\b', r'\b\w*years\w*\b'
    ]
    
    keywords = []
    for pattern in important_patterns:
        matches = re.findall(pattern, text)
        keywords.extend(matches)
    
    return list(set(keywords))[:12]

def find_matching_keywords(resume_text, job_keywords):
    """Find which keywords from job are present in resume with comprehensive matching"""
    resume_lower = preprocess_text(resume_text)
    # Also keep original resume text for exact matching
    resume_original = resume_text.lower()
    
    matched = []
    missing = []
    
    for keyword in job_keywords:
        keyword_clean = keyword.strip().lower()
        
        # Create multiple variants to check
        keyword_variants = [
            keyword_clean,
            keyword_clean.replace(' ', ''),
            keyword_clean.replace('-', ' '),
            keyword_clean.replace('_', ' '),
            keyword_clean.replace('.', ''),
            keyword_clean.replace('+', 'plus'),
            keyword_clean.replace('#', 'sharp'),
            keyword_clean.replace('&', 'and')
        ]
        
        # Also add word boundary checks for exact matches
        exact_patterns = [
            r'\b' + re.escape(keyword_clean) + r'\b',
            r'\b' + re.escape(keyword_clean.replace(' ', '')) + r'\b' if ' ' in keyword_clean else None,
            r'\b' + re.escape(keyword_clean.replace(' ', '-')) + r'\b' if ' ' in keyword_clean else None
        ]
        exact_patterns = [p for p in exact_patterns if p is not None]
        
        found = False
        
        # Check variants in processed text
        for variant in keyword_variants:
            if variant in resume_lower or variant in resume_original:
                matched.append(keyword.strip())
                found = True
                break
        
        # If not found, check with regex patterns for exact matches
        if not found:
            for pattern in exact_patterns:
                if re.search(pattern, resume_original):
                    matched.append(keyword.strip())
                    found = True
                    break
        
        if not found:
            missing.append(keyword.strip())
    
    return matched, missing

def calculate_semantic_similarity(resume_text, job_description):
    """Calculate deep semantic similarity using sentence transformers"""
    try:
        # Split texts into sentences for better semantic understanding
        resume_sentences = [s.strip() for s in re.split(r'[.!?]+', resume_text) if len(s.strip()) > 20]
        job_sentences = [s.strip() for s in re.split(r'[.!?]+', job_description) if len(s.strip()) > 20]
        
        if not resume_sentences or not job_sentences:
            # Fallback to full text comparison
            resume_embedding = semantic_model.encode([resume_text])
            job_embedding = semantic_model.encode([job_description])
        else:
            # Create embeddings for sentences
            resume_embeddings = semantic_model.encode(resume_sentences)
            job_embeddings = semantic_model.encode(job_sentences)
            
            # Get average embeddings
            resume_embedding = np.mean(resume_embeddings, axis=0).reshape(1, -1)
            job_embedding = np.mean(job_embeddings, axis=0).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        
        return float(similarity)
        
    except Exception as e:
        print(f"Error in semantic similarity calculation: {e}")
        return 0.5  # Neutral fallback

def calculate_content_overlap_score(resume_text, job_description):
    """Calculate how much of the job requirements are covered in the resume"""
    try:
        # Extract key requirements and responsibilities from job description
        job_requirements = extract_job_requirements(job_description)
        resume_content = preprocess_text(resume_text)
        
        if not job_requirements:
            return 0.5
        
        coverage_score = 0
        total_weight = 0
        
        for requirement, weight in job_requirements:
            requirement_words = requirement.lower().split()
            requirement_coverage = 0
            
            # Check how well this requirement is covered in resume
            for word in requirement_words:
                if word in resume_content:
                    requirement_coverage += 1
            
            if requirement_words:
                requirement_score = requirement_coverage / len(requirement_words)
                coverage_score += requirement_score * weight
                total_weight += weight
        
        return coverage_score / total_weight if total_weight > 0 else 0.5
        
    except Exception as e:
        print(f"Error in content overlap calculation: {e}")
        return 0.5

def extract_job_requirements(job_description):
    """Extract weighted requirements from job description"""
    requirements = []
    text_lower = job_description.lower()
    
    # High priority indicators
    high_priority_patterns = [
        r'required?:?\s*([^.!?\n]+)',
        r'must have:?\s*([^.!?\n]+)',
        r'essential:?\s*([^.!?\n]+)',
        r'minimum qualifications?:?\s*([^.!?\n]+)'
    ]
    
    # Medium priority indicators
    medium_priority_patterns = [
        r'preferred:?\s*([^.!?\n]+)',
        r'desired:?\s*([^.!?\n]+)',
        r'nice to have:?\s*([^.!?\n]+)',
        r'responsibilities?:?\s*([^.!?\n]+)'
    ]
    
    # Extract high priority requirements
    for pattern in high_priority_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match.strip()) > 10:
                requirements.append((match.strip(), 1.0))
    
    # Extract medium priority requirements
    for pattern in medium_priority_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if len(match.strip()) > 10:
                requirements.append((match.strip(), 0.6))
    
    # If no structured requirements found, extract key sentences
    if not requirements:
        sentences = [s.strip() for s in re.split(r'[.!?]+', job_description) if len(s.strip()) > 30]
        for sentence in sentences[:10]:  # Take first 10 meaningful sentences
            requirements.append((sentence, 0.3))
    
    return requirements[:15]  # Limit to top 15 requirements

def calculate_advanced_similarity(resume_text, job_description, matched_keywords, missing_keywords):
    """Calculate comprehensive similarity using multiple advanced methods"""
    try:
        print("=== CALCULATING ADVANCED SIMILARITY ===")
        
        # Method 1: Semantic similarity using sentence transformers (40% weight)
        semantic_sim = calculate_semantic_similarity(resume_text, job_description)
        print(f"Semantic similarity: {semantic_sim:.3f}")
        
        # Method 2: Content overlap analysis (30% weight)
        content_overlap = calculate_content_overlap_score(resume_text, job_description)
        print(f"Content overlap: {content_overlap:.3f}")
        
        # Method 3: TF-IDF similarity (15% weight)
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1000, stop_words='english')
            vectors = vectorizer.fit_transform([resume_text, job_description])
            tfidf_sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        except:
            tfidf_sim = 0.5
        print(f"TF-IDF similarity: {tfidf_sim:.3f}")
        
        # Method 4: Dynamic keyword overlap (15% weight)
        total_keywords = len(matched_keywords) + len(missing_keywords)
        if total_keywords > 0:
            keyword_overlap = len(matched_keywords) / total_keywords
        else:
            keyword_overlap = 0.5
        print(f"Keyword overlap: {keyword_overlap:.3f}")
        
        # Weighted combination
        final_similarity = (
            semantic_sim * 0.60 +
            content_overlap * 0.20 +
            tfidf_sim * 0.10 +
            keyword_overlap * 0.10
        ) * 100
        
        # Ensure reasonable range with dynamic scaling
        if final_similarity < 20:
            final_similarity = 20 + (final_similarity * 0.5)
        elif final_similarity > 95:
            final_similarity = 95
        
        print(f"Final similarity score: {final_similarity:.1f}%")
        print("=======================================")
        
        return final_similarity
        
    except Exception as e:
        print(f"Error in advanced similarity calculation: {e}")
        return 45.0  # Fallback score

async def calculate_match(resume_text, job_description):
    """Enhanced matching calculation with semantic analysis and AI-powered suggestions"""
    
    print("=== STARTING ENHANCED MATCH CALCULATION ===")
    
    # Validate inputs
    if not resume_text or not job_description:
        raise ValueError("Resume text and job description cannot be empty")
    
    if len(resume_text.strip()) < 50:
        raise ValueError("Resume text is too short for proper analysis")
        
    if len(job_description.strip()) < 50:
        raise ValueError("Job description is too short for proper analysis")
    
    print(f"Resume text length: {len(resume_text)}")
    print(f"Job description length: {len(job_description)}")
    
    # Extract dynamic keywords using Gemini
    print("=== EXTRACTING DYNAMIC KEYWORDS ===")
    job_keywords = await extract_dynamic_keywords(job_description)
    print(f"Extracted {len(job_keywords)} critical keywords: {job_keywords}")
    
    # Find matching and missing keywords
    matched_keywords, missing_keywords = find_matching_keywords(resume_text, job_keywords)
    
    # Calculate advanced similarity score using multiple methods
    similarity_score = calculate_advanced_similarity(resume_text, job_description, matched_keywords, missing_keywords)
    
    print(f"Similarity score: {similarity_score:.1f}%")
    print(f"Matched keywords: {matched_keywords}")
    print(f"Missing keywords: {missing_keywords}")
    
    # Create comprehensive prompt for Gemini
    prompt = create_detailed_prompt(job_description, resume_text, similarity_score, matched_keywords, missing_keywords)
    
    # Get AI-powered suggestions with retry mechanism
    max_retries = 3
    suggestions = None
    
    for attempt in range(max_retries):
        try:
            print(f"=== GEMINI ATTEMPT {attempt + 1}/{max_retries} ===")
            suggestions = get_gemini_response(prompt)
            
            if suggestions and len(suggestions.strip()) >= 200:
                print("✅ Gemini suggestions received successfully")
                break
            else:
                print(f"❌ Attempt {attempt + 1}: Insufficient suggestions length: {len(suggestions) if suggestions else 0}")
                if attempt < max_retries - 1:
                    print("Retrying with enhanced prompt...")
                    prompt = create_fallback_prompt(job_description, resume_text, similarity_score, matched_keywords, missing_keywords)
                
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
    
    # Final validation
    if not suggestions or len(suggestions.strip()) < 200:
        raise Exception("Failed to generate adequate AI suggestions after multiple attempts")

    print("=== MATCH CALCULATION COMPLETED ===")
    print(f"Final suggestions length: {len(suggestions)}")

    return {
        "similarity_score": round(similarity_score, 1),
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "suggestion": suggestions
    }

def create_detailed_prompt(job_description, resume_text, similarity_score, matched_keywords, missing_keywords):
    """Create a comprehensive and detailed prompt for Gemini analysis"""
    
    return f"""You are a world-class resume optimization expert, ATS specialist, and senior career strategist with over 20 years of experience in recruitment, talent acquisition, and career coaching. You have helped thousands of professionals optimize their resumes for Fortune 500 companies, startups, government positions, and specialized industries. You understand the intricacies of ATS systems, hiring manager psychology, recruitment trends, and what makes candidates stand out in competitive job markets.

=== COMPREHENSIVE ANALYSIS CONTEXT ===

TARGET JOB POSTING (COMPLETE):
{job_description}

CANDIDATE'S CURRENT RESUME (COMPLETE):
{resume_text}

=== DETAILED ANALYSIS METRICS ===
• Overall Match Score: {similarity_score:.1f}% (Based on semantic analysis, content overlap, TF-IDF similarity, and keyword matching)
• Successfully Matched Critical Keywords: {', '.join(matched_keywords) if matched_keywords else 'None identified'}
• Missing Critical Keywords: {', '.join(missing_keywords) if missing_keywords else 'None identified'}
• Keyword Match Rate: {len(matched_keywords)}/{len(matched_keywords) + len(missing_keywords)} critical requirements met

=== YOUR EXPERTISE AREAS TO CONSIDER ===
1. ATS Optimization: Understanding how Applicant Tracking Systems scan, parse, and rank resumes
2. Keyword Strategy: Knowing which keywords are deal-breakers vs. nice-to-have
3. Industry Knowledge: Understanding sector-specific terminology, trends, and expectations
4. Recruiter Psychology: Knowing what catches hiring managers' attention in the first 6 seconds
5. Competitive Analysis: Understanding what separates top candidates from average ones
6. Experience Positioning: How to frame experiences to maximize relevance and impact
7. Quantification Strategies: Which metrics matter most for different roles and industries
8. Skills Hierarchy: Understanding which technical and soft skills to prioritize and emphasize
9. Career Progression: How to position career gaps, transitions, or non-linear paths effectively
10. Personal Branding: Creating a cohesive narrative that differentiates the candidate

=== COMPREHENSIVE ANALYSIS INSTRUCTIONS ===

Analyze every aspect of this candidate's resume against the specific job requirements. Consider:

TECHNICAL ANALYSIS:
- Which specific technical skills, tools, programming languages, software, or platforms mentioned in the job are missing from the resume?
- How can the candidate better showcase their technical competency levels?
- What certifications, training, or projects would significantly strengthen their profile?
- Are there technical achievements or projects they should highlight more prominently?

EXPERIENCE ANALYSIS:
- How can they reframe their current experience to better align with the job requirements?
- What specific accomplishments should be quantified with numbers, percentages, or metrics?
- Which experiences are most relevant and should be emphasized?
- What language changes would make their impact more compelling?

INDUSTRY & ROLE ANALYSIS:
- What industry-specific terminology or jargon should they incorporate?
- How can they demonstrate understanding of the company's business model, challenges, or industry?
- What soft skills or leadership qualities are most valued in this specific role?
- How can they show progression and growth relevant to this position?

ATS & FORMATTING ANALYSIS:
- Which exact keywords from the job posting should be naturally integrated into their resume?
- How can they improve their resume's ATS compatibility without sacrificing readability?
- What sections or formats would improve their resume's scannability?

COMPETITIVE ADVANTAGE ANALYSIS:
- What unique value propositions can they highlight to stand out from other candidates?
- How can they demonstrate thought leadership or industry knowledge?
- What achievements or experiences give them a competitive edge?
- How can they show cultural fit and alignment with company values?

STRATEGIC POSITIONING ANALYSIS:
- How should they position their career story to make this role feel like a natural next step?
- What narrative threads should connect their past experiences to this opportunity?
- How can they address any obvious gaps or concerns preemptively?

=== OUTPUT REQUIREMENTS ===

Provide your analysis in EXACTLY this format (keep each section concise but impactful):

✨ STRENGTHS
• [Identify 1-2 specific strengths from their resume that directly relate to this job, with brief but impactful examples]
• [Mention specific achievements or experiences that make them competitive for this exact role]
• [Highlight unique qualifications or differentiators that set them apart from typical candidates]

🔍 IMPROVEMENTS NEEDED
Note: Only give 2-3 imporevements only under any circumstances, not more, and they should be the most impactful ones only and give them briefly.
• [If relevant projects, certifications, or technical demonstrations are missing, suggest adding them at priority]
• [Quantifiable metrics enhancement - suggest specific types of numbers, percentages, or results they should add with examples]
• [Experience positioning improvements - exactly how to reframe their background to better showcase relevant experience]

💡 PRO TIP
[Provide 2-3 sentences of advanced, strategic advice specific to this candidate and role. Include 2-3 exact keywords they must incorporate and one tactical positioning strategy that will significantly improve their competitiveness for this specific position.]

=== CRITICAL SUCCESS FACTORS ===
- Every suggestion must be immediately actionable and specific to THIS job and THIS candidate
- Reference actual content from their resume and the job posting
- Provide concrete examples rather than generic advice
- Focus on changes that will have the highest impact on their match score and interview chances
- Ensure all advice is realistic and implementable within their current experience level
- Prioritize suggestions that address the most critical gaps between their profile and job requirements

Your recommendations should transform this resume from its current state into one that would make any recruiter or hiring manager excited to interview this candidate for this specific role."""

def create_fallback_prompt(job_description, resume_text, similarity_score, matched_keywords, missing_keywords):
    """Create an enhanced fallback prompt if first attempt fails"""
    
    return f"""As an expert resume optimization consultant and ATS specialist, provide targeted recommendations to improve this candidate's match for the specific job posting.

DETAILED JOB REQUIREMENTS:
{job_description[:2000]}

CANDIDATE'S CURRENT RESUME:
{resume_text[:2000]}

ANALYSIS RESULTS:
- Current Match Score: {similarity_score:.1f}%
- Keywords Successfully Matched: {', '.join(matched_keywords) if matched_keywords else 'None'}
- Critical Missing Keywords: {', '.join(missing_keywords) if missing_keywords else 'None'}

TASK: Provide specific, actionable optimization advice in this exact format:

✨ STRENGTHS
• [1-2 specific strengths from their resume that directly relate to this job with brief examples]

🔍 IMPROVEMENTS NEEDED  
• [2-3 specific, actionable improvements that will most impact their chances]

💡 PRO TIP
[Strategic advice for this specific role and candidate. Include exact keywords to incorporate and tactical positioning advice in 2-3 sentences.]

Focus on practical changes they can implement immediately to significantly improve their match score and competitiveness for this exact position. Every suggestion must be specific to their background and this job's requirements."""