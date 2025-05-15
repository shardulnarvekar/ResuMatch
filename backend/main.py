from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.match import router as match_router


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to ResuMatch API"}

# Include your /api endpoints
app.include_router(match_router, prefix="/api")

# CORS for frontend-backend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


