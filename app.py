
from fastapi import FastAPI
from pydantic import BaseModel

from marine_functions import (
    marine_agent,
    generate_mcqs,
    generate_mock_test,
    oral_questions,
    interview_questions
)

app = FastAPI(title="Marine AI Agent")


class QuestionRequest(BaseModel):
    question: str


class TopicRequest(BaseModel):
    topic: str


@app.get("/")
def home():
    return {
        "message": "Marine AI API Running"
    }


@app.post("/chat")
def chat(request: QuestionRequest):
    return {
        "answer": marine_agent(request.question)
    }


@app.post("/mcq")
def mcq(request: TopicRequest):
    return {
        "mcq": generate_mcqs(request.topic)
    }


@app.post("/mock-test")
def mock_test(request: TopicRequest):
    return {
        "mock_test": generate_mock_test(request.topic)
    }


@app.post("/interview")
def interview(request: TopicRequest):
    return {
        "interview": interview_questions(request.topic)
    }


@app.post("/oral")
def oral(request: TopicRequest):
    return {
        "oral": oral_questions(request.topic)
    }
