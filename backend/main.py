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
INDEX_MAPPING_FILE = "./index_mapping.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

try:
    from logger import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    question: str
    # index_id: str

# --- Helper Functions ---

def load_index_mapping():
    """Loads the index_id to filename mapping from a JSON file."""
    if os.path.exists(INDEX_MAPPING_FILE):
        with open(INDEX_MAPPING_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode {INDEX_MAPPING_FILE}. Starting with empty mapping.")
                return {}
    return {}

def save_index_mapping(mapping):
    """Saves the index_id to filename mapping to a JSON file."""
    with open(INDEX_MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=4)

# --- PDF Processing Functions ---

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
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector store for index_id '{index_id}' not found at '{index_path}'")
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

# --- Endpoints ---

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        text = get_pdf_text(pdf_path)
        chunks = get_text_chunks(text)
        index_id = str(uuid.uuid4())
        save_vector_store(chunks, index_id)

        mapping = load_index_mapping()
        mapping[index_id] = file.filename
        save_index_mapping(mapping)

        logger.info(f"Created index: {index_id} for {file.filename}")
        return {"message": "PDF processed successfully", "index_id": index_id, "file_name": file.filename}
    except Exception as e:
        logger.error(f"Error during PDF upload and processing: {e}")
        return {"message": f"Error processing PDF: {e}", "status": "error"}, 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@app.post("/ask-question/")
def ask_question(req: QuestionRequest):
    try:
        vs = load_vector_store(req.index_id)
        docs = vs.similarity_search(req.question)
        chain = get_conversational_chain()
        result = chain.invoke({"context": docs, "question": req.question})
        return {"answer": result}
    except FileNotFoundError as e:
        logger.error(f"Error loading vector store for index_id {req.index_id}: {e}")
        return {"answer": f"Error: Vector store not found for selected index. Please re-upload or select a valid PDF. ({e})"}, 404
    except Exception as e:
        logger.error(f"Error processing question for index_id {req.index_id}: {e}")
        return {"answer": f"An error occurred while answering your question: {e}", "status": "error"}, 500


@app.post("/generate-flashcards/")
def generate_flashcards(req: QuestionRequest):
    try:
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
            logger.error(f"JSON decoding error for flashcards (index: {req.index_id}, topic: {req.question}): {e}, Raw: {raw}")
            return {"error": f"Could not parse flashcards from AI response. {e}", "raw_response": raw}

        return {"flashcards": flashcards}
    except FileNotFoundError as e:
        logger.error(f"Error loading vector store for index_id {req.index_id}: {e}")
        return {"error": f"Vector store not found for selected index. Please re-upload or select a valid PDF. ({e})"}, 404
    except Exception as e:
        logger.error(f"Error generating flashcards for index_id {req.index_id}: {e}")
        return {"error": f"An error occurred while generating flashcards: {e}"}, 500


@app.post("/generate-mcqs/")
def generate_mcqs(req: QuestionRequest):
    try:
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
            logger.error(f"JSON decoding error for MCQs (index: {req.index_id}, topic: {req.question}): {e}, Raw: {raw}")
            return {"error": f"Could not parse MCQs from AI response. {e}", "raw_response": raw}

        return {"mcqs": mcqs}
    except FileNotFoundError as e:
        logger.error(f"Error loading vector store for index_id {req.index_id}: {e}")
        return {"error": f"Vector store not found for selected index. Please re-upload or select a valid PDF. ({e})"}, 404
    except Exception as e:
        logger.error(f"Error generating MCQs for index_id {req.index_id}: {e}")
        return {"error": f"An error occurred while generating MCQs: {e}"}, 500


@app.get("/list-indexes/")
def list_indexes():
    stored_mapping = load_index_mapping()
    available_indexes_on_disk = [name for name in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, name))]

    result_indexes = []
    for index_id, file_name in stored_mapping.items():
        if index_id in available_indexes_on_disk:
            result_indexes.append({"id": index_id, "name": file_name})
        else:
            logger.warning(f"Index ID {index_id} found in mapping but not on disk. Skipping.")
    
    logger.info(f"Listing available indexes: {result_indexes}")
    return {"indexes": result_indexes}


@app.delete("/delete-index/{index_id}")
def delete_index(index_id: str):
    index_path = os.path.join(INDEX_DIR, index_id)
    if os.path.exists(index_path):
        import shutil
        shutil.rmtree(index_path)

        mapping = load_index_mapping()
        if index_id in mapping:
            del mapping[index_id]
            save_index_mapping(mapping)
        
        logger.info(f"Deleted index: {index_id}")
        return {"message": f"Index '{index_id}' and its mapping successfully deleted."}
    else:
        mapping = load_index_mapping()
        if index_id in mapping:
            del mapping[index_id]
            save_index_mapping(mapping)
            logger.warning(f"Index ID {index_id} not found on disk, but removed from mapping.")
            return {"message": f"Index '{index_id}' directory not found, but removed from mapping if existed."}
        
        logger.warning(f"Attempted to delete non-existent index: {index_id}")
        return {"message": f"Index '{index_id}' not found.", "status": "error"}, 404