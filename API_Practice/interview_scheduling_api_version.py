# This file is created just for practicing building APIs
# Created on 26-06-2025 at 18:22

import os
import warnings
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Bot Audio
from gtts import gTTS
import base64

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from typing import Optional, Union, Literal
from pydantic import BaseModel, Field, EmailStr
from functools import lru_cache

# SQL
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# SQL Connection
def create_connection():
    print("Creating Connection with DB")
    try:
        user = os.getenv("DB_USER")
        raw_password = os.getenv("DB_PASSWORD")
        password = quote_plus(raw_password)
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        db = os.getenv("DB_NAME")

        # Credentials of mySQL connection
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
        engine = create_engine(connection_string)
        print("Connection created Successfully")
        return engine
    except Exception as e:
        print(f"Error creating connection with DB: {e}")
        return None

# Input Schema Using Pydantic
    # For Interview Scheduling
class ScheduleInterviewInput(BaseModel):
    role:str = Field(description="Target Job Role")
    resume_path:str = Field(description="Path to resume file (PDF/DOCX/TXT)")
    question_limit:int = Field(description="Number of interview questions to generate")
    sender_email:EmailStr = Field(description="Sender's email address")

    # For Tracking Candidate
class TrackCandidateInput(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[EmailStr] = Field(None, description="Email address of the candidate")
    role: Optional[str] = Field(None, description="Role applied for, e.g., 'frontend', 'backend'")
    date_filter: Optional[str] = Field(
        None,
        description="Optional date filter: 'today', 'recent', or 'last_week'"
    )
    status: Optional[Literal["Scheduled", "Completed"]] = None

# For Current Day
current_month_year = datetime.now().strftime("%B %Y")

# Model
@lru_cache(maxsize=1)
def get_llm():
    return ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

llm = get_llm()

# Parser
parser = StrOutputParser()

# Prompt
prompt = PromptTemplate(
    template="""
You are an intelligent assistant that only answers questions based on the provided document content.

The document may include:
- Headings, paragraphs, subheadings
- Lists or bullet points
- Tables or structured data
- Text from PDF, DOCX, or TXT formats

Your responsibilities:
1. Use ONLY the content in the document to answer.
2. If the user greets (e.g., "hi", "hello"), respond with a friendly greeting.
3. Otherwise, provide a concise and accurate answer using only the document content.

Document Content:
{context}

User Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# Intent Prompt
intent_prompt = PromptTemplate(
    template="""You are an AI Intent Classifier for the Jobma Interviewing Platform. Based on the user input, identify their intent from the list of predefined intents.

Possible Intents:
- **schedule_interview**: The user wants to schedule an interview, usually mentions a role, resume, job title, or similar context.
- **track_candidate**: The user wants to check or track candidate interview details. This may include:
  - Asking for a specific candidate's status using an email or name.
  - Requesting a summary, details or list of all candidates interviewed.
  - Asking how many interviews have been conducted or who has been interviewed.
  - Asking for scheduled interviews (phrases like "show scheduled", "track scheduled", "upcoming interviews")
  - Asking for completed interviews (phrases like "show completed", "past interviews")
- **greet**: The user says hello, hi, good morning, or other greeting-like phrases.
- **help**: The user is asking for help or support about using the Jobma platform.
- **list_roles**: The user wants to view a list of roles interviews are scheduled for.
- **bye**: The user says goodbye or ends the conversation.
- **irrelevant**: The user input is unrelated to the Jobma platform or job interviews, such as asking about food, weather, sports, or general unrelated queries (e.g., "I want to make a pizza").

Classify the following user input strictly as one of the intents above. Your response must be a **single word** from the list: `schedule_interview`, `greet`, `help`, `bye`, or `irrelevant`.

User Input:
"{input}"

Intent:
""",
    input_variables=['input']
)

# Parsing Prompt
parsing_prompt = PromptTemplate(
    template="""
You are a helpful assistant that extracts filters to track a candidate's interview information.
Based on the user's request, extract and return a JSON object with the following keys:

- name: Candidate's name (if mentioned, like "Priya Sharma", "Dushyant Goyal")
- email: Candidate's email (e.g., "abc@example.com", "SinghDeepanshu1233@gmail.com")
- role: Role mentioned (like "backend", "frontend", "data analyst", "AI associate", etc.)
- date_filter: One of: "today", "recent", "last_week", or null if not mentioned
- status: "Scheduled" or "Completed" if mentioned (e.g., "show scheduled interviews" → "Scheduled")

Special cases:
- If user asks for "scheduled" or "upcoming" interviews, set status to "Scheduled"
- If user asks for "completed" or "past" interviews, set status to "Completed"

Only include relevant values. If a value is not mentioned, return null.

Input: {input}
Output:
""",
    input_variables=["input"]
)

# Document Loader
def extract_document(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError("Unsupported file format. Please upload a PDF, DOCX or TXT file.")

        docs = loader.load()
        print(f'Loading {file_path} ......')
        print('Loading Successful')
        return docs

    except FileNotFoundError as fe:
        print(f"File not found: {fe}")
    except Exception as e:
        print(f"Error loading document: {e}")
    
    return []

# RAG Workflow
def create_rag_chain(doc, prompt, parser, score_threshold=1.0, resume_text=False):
    # Document Loader
    docs = extract_document(doc)
    if not docs:
        raise ValueError("Document could not be loaded.")
    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Model Embeddings and Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embedding_model)
    # Retriever for Confidence Score
    retriever = vector_store.similarity_search_with_score

    def retrieve_using_confidence(query):
        results = retriever(query)
        filtered = [doc for doc, score in results if score <= score_threshold]
        return filtered

    def format_docs(retrieved_docs):
        if not retrieved_docs:
            return "INSUFFICIENT CONTEXT"
        return "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    parallel_chain = RunnableParallel({
        'context': RunnableLambda(lambda q: retrieve_using_confidence(q)) | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser
    if resume_text:
        return main_chain, "\n\n".join([doc.page_content for doc in docs])
    return main_chain