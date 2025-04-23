from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import json
from dotenv import load_dotenv


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from pydantic import BaseModel

load_dotenv()

GOOGLE_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_KEY")

class Question(BaseModel):
    question: str

def get_pdf_text(pdf_path):
    if os.path.isfile(pdf_path):
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    else:
        raise FileNotFoundError(f"No such file: '{pdf_path}'")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_GEMINI_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, just say "answer is not available in the context". 
    Do not make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_GEMINI_KEY
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return create_stuff_documents_chain(llm=model, prompt=prompt)

def ask_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_GEMINI_KEY
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain.invoke({
        "context": docs,
        "question": user_question
    })

    print("Raw response from chain:", response)
    return response

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173", "*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = {"*"},
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF QA API!"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    upload_dir = "./uploads"
    os.makedirs(upload_dir, exist_ok=True)

    pdf_path = os.path.join(upload_dir, file.filename)
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    text = get_pdf_text(pdf_path)
    chunks = get_text_chunks(text)
    get_vector_store(chunks)

    return {"message": "PDF processed successfully."}

@app.post("/ask-question/")
def ask_question_route(q: Question):
    response = ask_question(q.question)

    print("Full chain response:", response)

    if isinstance(response, dict) and "answer" in response:
        return {"answer": response["answer"]}
    elif isinstance(response, str):
        return {"answer": response}
    else:
        return {"answer": "Could not generate answer."}

@app.post("/generate-flashcards/")
def generate_flashcards(topic: Question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_GEMINI_KEY
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(topic.question)

    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Based on the following context, generate 7 educational flashcards on the topic: "{topic.question}".

    Context:
    {context_text}

    Format:
    Return only a JSON array of flashcards, each with 'question' and 'answer' keys.
    Return only a raw JSON array. Do NOT wrap it with triple backticks or any markdown.
    Do not include any markdown, triple quotes, or extra commentary.

    Example:
    [
        {{
            "question": "What is DNS?",
            "answer": "DNS stands for Domain Name System..."
        }},
        ...
    ]
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        google_api_key=GOOGLE_GEMINI_KEY
    )

    response = model.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

    print("cleaned flashcard response:", cleaned)

    try:
        if isinstance(raw, str):
            flashcards = json.loads(cleaned)
        elif isinstance(raw, list):
            flashcards = raw
        else:
            raise ValueError("Unexpected response format")
    except Exception as e:
        return {
            "error": "Failed to parse flashcards.",
            "details": str(e),
            "raw_response": str(raw)
        }

    return {"flashcards": flashcards}

@app.post("/generate-mcqs/")
def generate_mcqs(topic: Question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_GEMINI_KEY
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(topic.question)

    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Based on the following context, generate 7 multiple choice questions (MCQs) on the topic: "{topic.question}".

    Each question should have:
    - One question stem
    - Four options labeled A, B, C, and D
    - The correct answer indicated as a letter (A/B/C/D)

    Return only a raw JSON array in the following format:

    [
        {{
            "question": "What does DNS stand for?",
            "options": {{
                "A": "Dynamic Network Service",
                "B": "Domain Name System",
                "C": "Distributed Naming Service",
                "D": "Data Network Structure"
            }},
            "correct_answer": "B"
        }},
        ...
    ]

    Do not add any markdown, quotes, or explanation—just return raw JSON.
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        google_api_key=GOOGLE_GEMINI_KEY
    )

    response = model.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

    try:
        if isinstance(raw, str):
            mcqs = json.loads(cleaned)
        elif isinstance(raw, list):
            mcqs = raw
        else:
            raise ValueError("Unexpected response format")
    except Exception as e:
        return {
            "error": "Failed to parse MCQs.",
            "details": str(e),
            "raw_response": str(raw)
        }

    return {"mcqs": mcqs}
