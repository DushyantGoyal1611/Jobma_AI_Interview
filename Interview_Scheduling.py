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
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.agents import Tool, initialize_agent, AgentType
# Memory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# LLM to be used
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# For Current Day
current_month_year = datetime.now().strftime("%B %Y")

# Target Role
target_role = input("Target Role: ").strip()

if not target_role:
    print("Target Role not provided. Exiting .....")
    exit()

# Question Limit
while True:
    try:
        question_limit = int(input("How many Questions you want?: "))
        if question_limit > 0:
            break
        else:
            print("Please enter a number greater than 0.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

# Output Parser
parser = JsonOutputParser()

# Skills and Experience fetching prompt
resume_prompt = PromptTemplate(
    template="""
You are an AI Resume Analyzer. Analyze the resume text below and extract **only** information relevant to the given job role.

Your output **must** be in the following JSON format:
{format_instruction}

**Instructions:**
1. **Name**:
   - Extract the candidate's full name from the **first few lines** of the resume.
   - It is usually the **first large bold text** or line that is **not an address, email, or phone number**.
   - Exclude words like "Resume", "Curriculum Vitae", "AI", or job titles.
   - If the name appears to be broken across lines, reconstruct it (e.g., "Dushy" and "ant" should be "Dushyant").
   - If no clear name is found, return: `"Name": "NA"`.

2. **Skills**:
   - Extract technical and soft skills relevant to the **target role**.
   - Exclude generic or irrelevant skills (e.g., MS Word, Internet Browsing).
   - If **no skills are relevant**, return an empty list: `"Skills": []`.

3. **Experience**:
   - Calculate the **cumulative time spent at each company** to get total professional experience.
   - Include only non-overlapping, clearly dated experiences (internships, jobs).
   - If a role ends in "Present" or "Current", treat it as ending in **{current_month_year}**.
   - Example: 
     - Google: Jan 2023 - Mar 2023 = 2 months  
     - Jobma: Feb 2025 - May 2025 = 3 months  
     - Total: 5 months = `"Experience": "0.42 years"`
   - Round the final answer to **2 decimal places**.
   - If durations are missing or unclear, return: `"Experience": "NA"`.

4. Fetch email id from the document
   - Extract the first valid email address ending with `@gmail.com` from the text.
   - If not found, return `"Email": "NA"`.

---

**Target Role**: {role}

**Resume Text**:
{context}
""",
    input_variables=["context", "role"],
    partial_variables={
        "format_instruction": parser.get_format_instructions(),
        "current_month_year": current_month_year
        }
)

def extract_document(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

        docs = loader.load_and_split()
        print(f'Loading {file_path} ......')
        print('Loading Successful')
        return docs
    except FileNotFoundError as fe:
        print(f'File not Found: {fe}')
    except Exception as e:
        print(f"Error loading document: {e}")

    return []

def rag_workflow(doc:str, prompt):
    # Document Loader
    docs = extract_document(doc)
    if not docs:
        raise ValueError("Error in Document Loading ....")
    
    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embedding Model and Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
        "role": lambda _: target_role 
        }
        | prompt
        | llm
        | parser
    )

    return rag_chain

main_chain = rag_workflow("Ashutosh_AI_Engineer.pdf", resume_prompt)

# Invoking
resume_result = main_chain.invoke("Extract name, skills, years of experience and mail id from this resume.")
print("Extracted: ",resume_result)

sender_email = str(input("Enter Sender's Email ID: "))

# Skills, Experience and Email Id
name = resume_result.get("Name", "NA")
skills = resume_result.get("Skills", [])
experience = resume_result.get("Experience", "NA")
email = resume_result.get("Email", "NA")

# Creating JSON File
interview_context = {
    "name" : name,
    "target_role" : target_role,
    "skills" : skills,
    "experience" : experience,
    "email" : email,
    "question_limit" : question_limit,
    "sender_email" : sender_email
}

json_file = "interview_context.json"

if os.path.exists(json_file):
    with open(json_file, "r") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            data = []
else:
    data = []

data.append(interview_context)

with open(json_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Interview context saved to '{json_file}'")