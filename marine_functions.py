
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

    prompt = f"""
Generate EXACTLY 5 Marine Engineering MCQs.

Marine Knowledge:
{context}

FORMAT:

Q1. Question?

A) Option A
B) Option B
C) Option C
D) Option D

Correct Answer: A

RULES:
- Start directly from Q1
- No introduction
- No explanations
- Every option on a new line
- Leave one blank line between questions
"""

    return llm.invoke(prompt).content


def generate_mock_test(topic):
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Create a Marine Engineering Mock Test.

Marine Knowledge:
{context}

Topic:
{topic}

FORMAT:

SECTION A: MCQs

Q1. Question?

A) Option A
B) Option B
C) Option C
D) Option D

Correct Answer: A

SECTION B: Short Questions

Q1. What is Purifier?

Answer:
Write answer here.

SECTION C: Viva Questions

Q1. What is MARPOL?

Answer:
Write answer here.

RULES:

* Proper formatting
* No long paragraphs
* Leave blank lines between questions
  """

    return llm.invoke(prompt).content


def oral_questions(topic):
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Generate exactly 20 Marine Engineering Oral/Viva Questions.

Marine Notes:
{context}

FORMAT:

Q1. What is SOLAS?

Answer:
SOLAS stands for Safety of Life at Sea.

Q2. What is MARPOL?

Answer:
MARPOL is an international convention.

RULES:

* Question on one line
* Answer on next line
* Leave one blank line between questions
* No introduction
  """

    return llm.invoke(prompt).content


def interview_questions(topic):
    docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Generate exactly 20 Marine Engineering Interview Questions.

Marine Notes:
{context}

FORMAT:

Q1. What is a Crosshead Engine?

Answer:
A Crosshead Engine is a large marine diesel engine.

Q2. What is a Purifier?

Answer:
A Purifier separates water and impurities.

RULES:

* Question on one line
* Answer on next line
* Leave one blank line between questions
* No introduction
  """

    return llm.invoke(prompt).content
