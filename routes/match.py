from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from utils.resume_parser import extract_resume_text
from utils.matcher_tf import ResuMatchTensorflow
import json

router = APIRouter()
matcher = ResuMatchTensorflow()

@router.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        contents = await file.read()
        resume_text = extract_resume_text(contents, file.filename)
        
        # Use the TensorFlow matcher instead of the old matcher
        result = matcher.calculate_match(resume_text, job_description)
        
        return result
    except Exception as e:
        print("ERROR:", str(e))  # prints in terminal
        raise HTTPException(status_code=500, detail=str(e))
        
@router.post("/suggest-improvements")
async def suggest_improvements(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    try:
        contents = await file.read()
        resume_text = extract_resume_text(contents, file.filename)
        
        # Use OpenAI API to generate improvement suggestions
        from utils.openai_helper import get_resume_improvement_suggestions
        suggestions = get_resume_improvement_suggestions(resume_text, job_description)
        
        return {"suggestions": suggestions}
    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))