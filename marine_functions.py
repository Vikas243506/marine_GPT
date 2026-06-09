
import os

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="marine_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def marine_agent(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a Marine Engineering Tutor.

Answer ONLY from provided knowledge.

{context}

Question: {question}
"""

    return llm.invoke(prompt).content


def generate_mcqs(topic):
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Generate 5 MCQs from:
{context}
"""

    return llm.invoke(prompt).content


def generate_mock_test(topic):
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Create mock test:
{context}
Topic: {topic}
"""

    return llm.invoke(prompt).content


def oral_questions(topic):
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Generate 20 oral questions:
{context}
"""

    return llm.invoke(prompt).content


def interview_questions(topic):
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Generate 20 interview questions:
{context}
"""

    return llm.invoke(prompt).content
