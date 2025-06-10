# This is the Interviewer's Side

import os
import re
import json
import warnings
import streamlit as st
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

# Streamlit for User Interface
st.set_page_config(page_title="AI Interview Assistant", layout="centered")
st.title("AI Interview Assistant")

# Pydantic Model
class ScheduleInterviewInput(BaseModel):
    role:str = Field(description="Target Job Role")
    resume_path:str = Field(description="Path to resume file (PDF/DOCX/TXT)")
    question_limit:int = Field(description="Number of interview questions to generate")
    sender_email: str = Field(description="Sender's email address")

# For Current Day
current_month_year = datetime.now().strftime("%B %Y")

# Model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

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
        st.write(f'Loading {file_path} ......')
        st.write('Loading Successful')
        return docs

    except FileNotFoundError as fe:
        st.error(f"File not found: {fe}")
    except Exception as e:
        st.error(f"Error loading document: {e}")
    
    return []

# RAG WorkFlow
def create_rag_chain(doc, prompt, parser, resume_text=False):
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
    # Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

    def format_docs(retrieved_docs):
        return "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()    
    })

    main_chain = parallel_chain | prompt | llm | parser
    if resume_text:
        return main_chain, "\n\n".join([doc.page_content for doc in docs])
    return main_chain

# Skills and Experience fetching prompt and JSON Parser
resume_parser = JsonOutputParser()

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
   - If the name appears to be broken across lines, reconstruct it (e.g., "Abhis" and "hek" should be "Abhishek").
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
        "format_instruction": resume_parser.get_format_instructions(),
        "current_month_year": current_month_year
    }
)

def schedule_interview(role:str|dict, resume_path:str, question_limit:int, sender_email:str) -> str:
    """
    Schedules interview by extracting details from resume and saving context to JSON.
    Args:
        role: Target role
        resume_path: Path to the candidate's resume file
        question_limit: Number of questions to generate
        sender_email: Sender's email
    Returns:
        Confirmation string
    """
    if not isinstance(resume_path, str):
        raise ValueError("resume_path must be a valid string")

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume file not found at path: {resume_path}")

    if isinstance(role, dict):
        role = role.get("title", "")
        if not role:
            role = str(role) 
    
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Role must be a non-empty string")

    # Create the chain with JsonOutputParser instead of StrOutputParser
    resume_chain = resume_prompt | llm | resume_parser
    resume_result = resume_chain.invoke({'context': extract_document(resume_path), 'role': role})

    interview_context = {
        "name": resume_result.get("Name", "NA"),
        "target_role": role,
        "skills": resume_result.get("Skills", []),
        "experience": resume_result.get("Experience", "NA"),
        "email": resume_result.get("Email", "NA"),
        "question_limit": question_limit,
        "sender_email": sender_email
    }

    json_file = "interview_context.json"
    data = []

    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        except json.JSONDecodeError:
            data = []

    data.append(interview_context)

    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)

    return f"Interview scheduled successfully and saved to '{json_file}'."

# Agent
interview_tool = StructuredTool.from_function(
    schedule_interview,
    args_schema=ScheduleInterviewInput,
    name="schedule_interview",
    description="Extracts resume information and schedules interview"
)

# The Chatbot
# In the ask_ai() function, replace the relevant sections with:

def ask_ai():
    st.subheader("Ask the AI or Schedule an Interview")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize document chain as None
    chain = None
    
    # Document upload section
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Uploaded file: {uploaded_file.name}")
        try:
            chain = create_rag_chain(file_path, prompt, parser)
        except Exception as e:
            st.error(f"Error creating RAG chain: {str(e)}")
            chain = None

    # Agent tool setup
    tools = [interview_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    fallback_triggers = r"(insufficient|not (sure|enough|understand)|i don't know|no context)"

    # User Input Area
    user_input = st.text_input("You:", placeholder="Type your question or 'schedule interview'")
    
    if user_input:
        if user_input.lower() == "schedule interview":
            st.info("Fill the form below to schedule an interview:")
            with st.form("interview_form"):
                role = st.text_input("Target Role")
                uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
                question_limit = st.number_input("Number of Questions", min_value=1, value=5)
                sender_email = st.text_input("Sender Email")

                submit = st.form_submit_button("Schedule")

                if submit and uploaded_resume is not None:
                    # Save the uploaded resume temporarily
                    temp_resume_path = os.path.join("temp_uploads", uploaded_resume.name)
                    with open(temp_resume_path, "wb") as f:
                        f.write(uploaded_resume.getbuffer())
                    
                    try:
                        response = schedule_interview(
                            role=role,
                            resume_path=temp_resume_path,
                            question_limit=question_limit,
                            sender_email=sender_email
                        )
                        st.success(f"AI (Agent): {response}")
                        st.session_state.chat_history.append(f"User: schedule interview for {role}")
                        st.session_state.chat_history.append(f"AI: {response}")
                    except Exception as e:
                        st.error(f"Error during Scheduling: {str(e)}")
                elif submit and uploaded_resume is None:
                    st.error("Please upload a resume file")

        else:
            if chain:
                try:
                    response = chain.invoke(user_input)
                except Exception as e:
                    response = f"Error processing your question: {str(e)}"
            else:
                response = "I don't have access to the knowledge base. Please ask general questions or schedule an interview."

            if re.search(fallback_triggers, response, re.IGNORECASE):
                st.write("Fallback Triggered: Using AI for external info... ")
                messages = [HumanMessage(content=user_input)]
                final_response = llm.invoke(messages)
                response = final_response.content
                
            st.session_state.chat_history.append(f"User: {user_input}")
            st.session_state.chat_history.append(f"AI: {response}")
            st.write(f"AI: {response}")

    # Display chat history
    st.markdown("---")
    st.subheader("Chat History")
    for msg in st.session_state.chat_history:
        # Split into role and content if needed
        if isinstance(msg, str) and ":" in msg:
            role, content = msg.split(":", 1)
            st.markdown(f"**{role.strip()}:** {content.strip()}")
        else:
            st.markdown(str(msg))

ask_ai()