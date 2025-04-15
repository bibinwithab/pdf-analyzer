# main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from pydantic import BaseModel

GOOGLE_GEMINI_KEY = 'AIzaSyARTnyuSl51LQNEuHKC5oGQU8l_A2WORsI'

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
    allow_origins=["http://localhost:5173"],  # or ["*"] for all origins (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF QA API!"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = f"./{file.filename}"
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    text = get_pdf_text(pdf_path)
    chunks = get_text_chunks(text)
    get_vector_store(chunks)

    return {"message": "PDF processed and vector store created."}

@app.post("/ask-question/")
@app.post("/ask-question/")
def ask_question_route(q: Question):
    response = ask_question(q.question)

    print("Full chain response:", response)

    # assuming LangChain returns dict
    if isinstance(response, dict) and "answer" in response:
        return {"answer": response["answer"]}
    elif isinstance(response, str):
        return {"answer": response}
    else:
        return {"answer": "Could not generate answer."}

