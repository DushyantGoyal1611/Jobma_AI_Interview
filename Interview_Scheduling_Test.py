# Improved Interviewer's Side

import os
import re
import json
import warnings
from datetime import datetime
from dotenv import load_dotenv

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

# Config
warnings.filterwarnings('ignore')
load_dotenv()

# Globals
current_month_year = datetime.now().strftime("%B %Y")
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Resume Input Schema
class ScheduleInterviewInput(BaseModel):
    role: str = Field(description="Target Job Role")
    resume_path: str = Field(description="Path to resume file (PDF/DOCX/TXT)")
    question_limit: int = Field(description="Number of interview questions to generate")
    sender_email: str = Field(description="Sender's email address")

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
2. If the question is clearly related to the document topic but the content is insufficient, respond with: INSUFFICIENT CONTEXT.
3. If the question is completely unrelated to the document, respond with: SORRY: This question is irrelevant.
4. If the user greets (e.g., "hi", "hello"), respond with a friendly greeting.
5. Otherwise, provide a concise and accurate answer using only the document content.

Document Content:
{context}

User Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# Document Loader

def extract_document(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format")
    return loader.load()

# Create RAG Chain

def create_rag_chain(file_path, prompt, parser):
    docs = extract_document(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})

    chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda x: "\n\n".join([d.page_content for d in x])),
        "question": RunnablePassthrough()
    }) | prompt | llm | parser

    return chain

# Resume Parsing Prompt
resume_parser = JsonOutputParser()
resume_prompt = PromptTemplate(
    template="""
You are an AI Resume Analyzer. Analyze the resume text below and extract only information relevant to the given job role.

Target Role: {role}

Output format:
{format_instruction}

Instructions:
- Extract name, relevant skills, experience (in years), and first @gmail.com email.
- Use {current_month_year} as current date for 'Present' roles.

Resume Text:
{context}
""",
    input_variables=["context", "role"],
    partial_variables={
        "format_instruction": resume_parser.get_format_instructions(),
        "current_month_year": current_month_year
    }
)

# Schedule Interview

def schedule_interview(role: str, resume_path: str, question_limit: int, sender_email: str) -> str:
    docs = extract_document(resume_path)
    resume_text = "\n\n".join([doc.page_content for doc in docs])
    chain = resume_prompt | llm | resume_parser
    result = chain.invoke({"context": resume_text, "role": role})

    context_data = {
        "name": result.get("Name", "NA"),
        "target_role": role,
        "skills": result.get("Skills", []),
        "experience": result.get("Experience", "NA"),
        "email": result.get("Email", "NA"),
        "question_limit": question_limit,
        "sender_email": sender_email
    }

    json_file = "interview_context.json"
    existing = []
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
        except json.JSONDecodeError:
            existing = []

    existing.append(context_data)
    with open(json_file, "w") as f:
        json.dump(existing, f, indent=2)

    return f"Interview scheduled and saved to '{json_file}'"

# Structured Tool
interview_tool = StructuredTool.from_function(
    schedule_interview,
    args_schema=ScheduleInterviewInput,
    name="schedule_interview",
    description="Extracts resume information and schedules interview"
)

# AI Assistant Chat

def ask_ai():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    parser = StrOutputParser()
    rag_chain = create_rag_chain("formatted_QA.txt", prompt, parser)

    tools = [interview_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

    fallback_triggers = re.compile(r"(insufficient|not (sure|enough|understand)|i don't know|no context)", re.IGNORECASE)

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting Chat.\nGoodbye!")
            break

        # Agent Call for Schedule Interview
        if user_input.lower().startswith("schedule interview"):
            print("Invoking Interview Scheduler Agent...")
            try:
                role = input("Enter Target Role: ")
                resume_path = input("Enter Resume Path: ")
                if not os.path.isfile(resume_path):
                    print(f"Error: File not found at {resume_path}")
                    continue
                question_limit = int(input("Question Limit: "))
                sender_email = input("Sender's Email: ")

                tool_input = {
                    "role": role,
                    "resume_path": resume_path,
                    "question_limit": question_limit,
                    "sender_email": sender_email
                }
                print("Agent Response:", agent.invoke({"input": f"schedule interview", **tool_input}))
                continue

            except Exception as e:
                print("Error during scheduling:", str(e))
                continue

        # RAG Response
        response = rag_chain.invoke(user_input)
        if fallback_triggers.search(response):
            print("Fallback Triggered. Using LLM for external info...")
            fallback_response = agent.invoke({"input": user_input})
            print(f"AI (Fallback): {fallback_response['output']}")
        else:
            print(f"AI: {response}")

ask_ai()