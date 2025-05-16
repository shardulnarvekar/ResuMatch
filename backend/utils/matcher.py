import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import re
from typing import List, Dict, Tuple, Set, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class ResuMatchTensorflow:
    def __init__(self):
        # Load Universal Sentence Encoder from TensorFlow Hub
        print("Loading Universal Sentence Encoder...")
        self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        print("Models loaded successfully!")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_keywords(self, text: str) -> Set[str]:
        """Extract and lemmatize important keywords from text"""
        # Clean the text
        clean = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(clean)
        
        # Filter out stop words and short words, then lemmatize
        keywords = set()
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemma = self.lemmatizer.lemmatize(token)
                keywords.add(lemma)
                
        return keywords

    def extract_skills_from_text(self, text: str) -> Set[str]:
        """Extract potential skills from text using NLP techniques"""
        # Common technical skills and qualifications (this can be expanded)
        common_skills = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node', 'express',
            'django', 'flask', 'fastapi', 'spring', 'html', 'css', 'sql', 'nosql', 'mongodb',
            'postgresql', 'mysql', 'oracle', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'terraform', 'jenkins', 'ci/cd', 'git', 'machine learning', 'deep learning',
            'tensorflow', 'pytorch', 'keras', 'nlp', 'computer vision', 'data science',
            'data analysis', 'agile', 'scrum', 'kanban', 'jira', 'confluence', 'leadership',
            'management', 'communication', 'teamwork', 'problem solving', 'critical thinking',
            'c++', 'c#', 'ruby', 'php', 'scala', 'swift', 'kotlin', 'rust', 'go', 'golang',
            'r', 'tableau', 'power bi', 'excel', 'word', 'powerpoint', 'sap', 'salesforce',
            'marketing', 'seo', 'sem', 'social media', 'content creation', 'adobe',
            'photoshop', 'illustrator', 'indesign', 'premiere', 'after effects',
            'accounting', 'finance', 'sales', 'customer service', 'human resources',
            'recruiting', 'operations', 'project management', 'product management',
            'ux/ui', 'user experience', 'user interface', 'responsive design',
            'mobile development', 'ios', 'android'
        }
        
        # Extract words and bi-grams from text
        clean = self.clean_text(text)
        tokens = word_tokenize(clean)
        
        # Create bigrams
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        
        # Check for skills in tokens and bigrams
        found_skills = set()
        for item in tokens + bigrams:
            if item.lower() in common_skills:
                found_skills.add(item.lower())
                
        return found_skills

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get text embeddings using Universal Sentence Encoder"""
        return self.use_model(texts).numpy()

    def calculate_semantic_similarity(self, resume_text: str, job_description: str) -> float:
        """Calculate semantic similarity using Universal Sentence Encoder"""
        # Get embeddings
        embeddings = self.get_embeddings([resume_text, job_description])
        resume_embedding = embeddings[0].reshape(1, -1)
        job_embedding = embeddings[1].reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        return similarity

    def calculate_match(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Calculate comprehensive match between resume and job description"""
        if not resume_text or not job_description:
            raise ValueError("Both resume text and job description are required")
            
        # Clean texts
        resume_clean = self.clean_text(resume_text)
        job_clean = self.clean_text(job_description)
        
        # 1. TF-IDF Similarity
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([resume_clean, job_clean])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # 2. Semantic Similarity with Universal Sentence Encoder
        semantic_similarity = self.calculate_semantic_similarity(resume_clean, job_clean)
        
        # 3. Keyword Matching
        resume_keywords = self.extract_keywords(resume_text)
        job_keywords = self.extract_keywords(job_description)
        matched_keywords = resume_keywords.intersection(job_keywords)
        missing_keywords = job_keywords - resume_keywords
        
        # 4. Skills Matching
        resume_skills = self.extract_skills_from_text(resume_text)
        job_skills = self.extract_skills_from_text(job_description)
        matched_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills - resume_skills
        
        # Calculate the overall score (weighted average)
        # You can adjust these weights based on importance
        tfidf_weight = 0.3
        semantic_weight = 0.5
        keyword_weight = 0.2
        
        # For keyword match ratio calculation
        keyword_match_ratio = len(matched_keywords) / max(len(job_keywords), 1)
        
        # Combined score
        overall_score = (
            tfidf_similarity * tfidf_weight +
            semantic_similarity * semantic_weight +
            keyword_match_ratio * keyword_weight
        ) * 100
        
        # Prepare detailed results
        result = {
            "overall_score": round(overall_score, 2),
            "tfidf_similarity": round(tfidf_similarity * 100, 2),
            "semantic_similarity": round(semantic_similarity * 100, 2),
            "keyword_match_ratio": round(keyword_match_ratio * 100, 2),
            "matched_keywords": sorted(list(matched_keywords)),
            "missing_keywords": sorted(list(missing_keywords)),
            "matched_skills": sorted(list(matched_skills)),
            "missing_skills": sorted(list(missing_skills))
        }
        
        # Generate suggestions based on missing skills and keywords
        suggestions = []
        
        if missing_skills:
            top_missing_skills = sorted(list(missing_skills))[:5]
            suggestions.append(f"Consider adding these skills: {', '.join(top_missing_skills)}")
            
        if missing_keywords and len(suggestions) < 3:
            top_missing_keywords = sorted(list(missing_keywords))[:5]
            suggestions.append(f"Add these keywords to better match the job: {', '.join(top_missing_keywords)}")
            
        result["suggestions"] = suggestions if suggestions else ["Your resume already matches well with this job description!"]
        
        return result


# Example usage
if __name__ == "__main__":
    matcher = ResuMatchTensorflow()
    
    # Example texts
    resume = """
    John Doe
    Software Engineer
    
    Experience:
    - 5 years of Python development
    - Built RESTful APIs using Django and Flask
    - Implemented machine learning models with TensorFlow
    - Git version control and CI/CD pipeline experience
    
    Skills:
    Python, JavaScript, SQL, Machine Learning, TensorFlow, Flask, Django
    """
    
    job = """
    Software Engineer Position
    
    Requirements:
    - Strong Python programming skills
    - Experience with web frameworks like Django or Flask
    - Knowledge of databases and SQL
    - Familiarity with Docker and Kubernetes
    - Git and version control experience
    
    Nice to have:
    - Machine learning experience
    - JavaScript and front-end development
    """
    
    result = matcher.calculate_match(resume, job)
    print(f"Overall Match Score: {result['overall_score']}%")
    print(f"TF-IDF Similarity: {result['tfidf_similarity']}%")
    print(f"Semantic Similarity: {result['semantic_similarity']}%")
    print(f"Matched Skills: {', '.join(result['matched_skills'])}")
    print(f"Missing Skills: {', '.join(result['missing_skills'])}")
    print(f"Suggestions: {result['suggestions']}")