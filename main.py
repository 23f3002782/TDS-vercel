from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load student marks from JSON file
with open('q-vercel-python.json', 'r') as f:
    marks_data = json.load(f)
    student_marks = {student['name']: student['marks'] for student in marks_data}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/api")
def get_marks(name: List[str]):
    marks = []
    for student_name in name:
        marks.append(student_marks.get(student_name, 0))
    return {"marks": marks}