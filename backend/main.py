from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import json
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from logger import logger

load_dotenv()
GOOGLE_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
INDEX_DIR = "./faiss_indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str
    # index_id: str

def get_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def save_vector_store(chunks, index_id):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_GEMINI_KEY
    )
    vs = FAISS.from_texts(chunks, embedding=embeddings)
    vs.save_local(os.path.join(INDEX_DIR, index_id))

def load_vector_store(index_id):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_GEMINI_KEY
    )
    index_path = os.path.join(INDEX_DIR, index_id)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    template = """
    Answer the question in markdown format.
    Context:
    {context}
    Question:
    {question}
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_GEMINI_KEY
    )
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return create_stuff_documents_chain(llm=llm, prompt=prompt)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    text = get_pdf_text(pdf_path)
    chunks = get_text_chunks(text)
    index_id = str(uuid.uuid4())
    save_vector_store(chunks, index_id)

    os.remove(pdf_path)
    logger.info(f"Created index: {index_id} for {file.filename}")
    return {"message": "PDF processed successfully", "index_id": index_id}

@app.post("/ask-question/")
def ask_question(req: QuestionRequest):
    vs = load_vector_store(req.index_id)
    docs = vs.similarity_search(req.question)
    chain = get_conversational_chain()
    result = chain.invoke({"context": docs, "question": req.question})
    return {"answer": result}

@app.post("/generate-flashcards/")
def generate_flashcards(req: QuestionRequest):
    vs = load_vector_store(req.index_id)
    docs = vs.similarity_search(req.question)
    context_text = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Based on the context, generate 7 flashcards for topic: "{req.question}".
    Context:
    {context_text}
    Return JSON array with 'question' and 'answer' keys. No markdown or explanation.
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        google_api_key=GOOGLE_GEMINI_KEY
    )
    raw = model.invoke(prompt).content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)

    try:
        flashcards = json.loads(cleaned)
    except Exception as e:
        return {"error": str(e), "raw_response": raw}

    return {"flashcards": flashcards}

@app.post("/generate-mcqs/")
def generate_mcqs(req: QuestionRequest):
    vs = load_vector_store(req.index_id)
    docs = vs.similarity_search(req.question)
    context_text = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Based on the context, generate 7 MCQs for topic: "{req.question}".
    Each should have question, 4 options (A-D), and correct_answer.
    Return raw JSON array only. No markdown or explanation.
    Context:
    {context_text}
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        google_api_key=GOOGLE_GEMINI_KEY
    )
    raw = model.invoke(prompt).content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)

    try:
        mcqs = json.loads(cleaned)
    except Exception as e:
        return {"error": str(e), "raw_response": raw}

    return {"mcqs": mcqs}

@app.get("/list-indexes/")
def list_indexes():
    indexes = [name for name in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, name))]
    return {"indexes": indexes}