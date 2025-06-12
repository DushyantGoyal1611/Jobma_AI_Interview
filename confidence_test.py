# This is the Interviewer's Side

import os
import re
import json
import warnings
from datetime import datetime
from dotenv import load_dotenv
# LangChain related libraries
# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# RAG Libraries
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.agents import initialize_agent, AgentType
# Memory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()


query = "How to track candidate's interview progress"

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

# Step 1: Load docs and create retriever (from your chain setup)
docs = extract_document('formatted_QA.txt')
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
vector_store = FAISS.from_documents(chunks, embedding_model)
retriever = vector_store.similarity_search_with_score

# Step 2: Query the retriever directly
results = retriever(query, k=3)

# Step 3: Print scores and contents
for i, (doc, score) in enumerate(results, 1):
    print(f"\nMatch {i}:")
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:300]}...")