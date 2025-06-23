# Interviewer Side 23-06-2025, 14:46
# Using streamlit side by side (will test on frontend instead of CLI)

import os, re
import warnings
import streamlit as st
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
from langchain_core.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from typing import Optional, Union
from pydantic import BaseModel, Field, EmailStr
from functools import lru_cache
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Streamlit for User Interface
st.set_page_config(page_title="AI Interview Assistant", layout="centered")
st.title("Jobma AI Interview Assistant")

st.sidebar.header("Settings")
use_voice = st.sidebar.checkbox("Enable Bot Voice", value=True)
st.session_state.use_voice = use_voice

# SQL Connection
@st.cache_resource
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

# Initialize only once
engine = create_connection()
if not engine:
    print("Database connection failed.")
    st.stop()

# Bot-Audio
def speak_text(text: str, lang="en") -> str:
    """Convert text to speech and return HTML audio player"""
    tts = gTTS(text, lang=lang)
    tts.save("bot_response.mp3")

    # Load and encode audio to base64
    with open("bot_response.mp3", "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()

    # Return an HTML audio player
    audio_html = f"""
    <audio controls autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

# Input Schema Using Pydantic
    # For Interview Scheduling
class ScheduleInterviewInput(BaseModel):
    role:str = Field(description="Target Job Role")
    resume_path:str = Field(description="Path to resume file (PDF/DOCX/TXT)")
    question_limit:int = Field(description="Number of interview questions to generate")
    sender_email:str = Field(description="Sender's email address")

    # For Tracking Candidate
class TrackCandidateInput(BaseModel):
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Email address of the candidate")
    role: Optional[str] = Field(None, description="Role applied for, e.g., 'frontend', 'backend'")
    date_filter: Optional[str] = Field(
        None,
        description="Optional date filter: 'today', 'recent', or 'last_week'"
    )

# For Current Day
current_month_year = datetime.now().strftime("%B %Y")

# LLM
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
            st.error("Unsupported file format. Please upload a PDF, DOCX or TXT file.")

        docs = loader.load()
        return docs

    except FileNotFoundError as fe:
        st.error(f"File not found: {fe}")
    except Exception as e:
        st.error(f"Error loading document: {e}")
    
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

5. **Phone**:
   - Extract the first 10-digit Indian mobile number (starting with 6-9) from the resume.
   - You can allow formats with or without `+91`, spaces, or dashes.
   - Examples: `9876543210`, `+91-9876543210`, `+91 98765 43210`.
   - If no valid number is found, return `"Phone": "NA"`.

6. **Education**:
   - Extract **highest qualification** (e.g., B.Tech, M.Tech, MCA, MBA, PhD).
   - Include the **degree name**, **specialization** (if available), and **university/institute name**.
   - Example: `"Education": "MCA in Computer Applications from VIPS, GGSIPU"`
   - If not found, return `"Education": "NA"`.
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

# Function to Schedule Interview
def schedule_interview(role:str|dict, resume_path:str, question_limit:int, sender_email:str) -> str:
    if not isinstance(resume_path, str):
        raise ValueError("resume_path must be a valid string")
    
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume file not found at path: {resume_path}")
    
    # Create the chain with JsonOutputParser instead of StrOutputParser
    resume_chain = resume_prompt | llm | resume_parser
    resume_result = resume_chain.invoke({'context': extract_document(resume_path), 'role': role})

    name = resume_result.get("Name", "NA")
    email = resume_result.get("Email", "NA")
    experience = resume_result.get("Experience", "NA")
    skills = ", ".join(resume_result.get("Skills", []))
    education = resume_result.get("Education", "NA")
    phone = resume_result.get("Phone", "NA")
    current_time = datetime.now()

    with engine.begin() as conn:  # Ensures transactional safety: commits on success, rolls back on error.
        # 1. Check if candidate exists
        result = conn.execute(text(
            "Select id from AI_INTERVIEW_PLATFORM.candidates Where email = :email"
        ),
        {"email": email}
        ).fetchone()

        if result:
            candidate_id = result[0]
        else:
            # 2. Insert candidate
            insert_candidate = text("""
                Insert into AI_INTERVIEW_PLATFORM.candidates (name, email, skills, education, experience, resume_path, phone, created_at)
                Values (:name, :email, :skills, :education, :experience, :resume_path, :phone, :created_at)
            """)
            conn.execute(insert_candidate, {
                "name": name,
                "email": email,
                "skills": skills,
                "education": education,
                "experience": experience,
                "resume_path": resume_path,
                "phone": phone,
                "created_at": current_time
            })

            # Get new candidate_id
                # Scalar fetches the first column of the first row of the result set, or returns None if there are no rows.
            candidate_id = conn.execute(
                text("SELECT id FROM AI_INTERVIEW_PLATFORM.candidates WHERE email = :email"),
                {"email": email}
            ).scalar()

        # 3. Insert interview_invitation
        insert_invite = text("""
            INSERT INTO AI_INTERVIEW_PLATFORM.interview_invitation 
                (candidate_id, role, question_limit, sender_email, status, created_at, interview_scheduling_time)
            VALUES 
                (:candidate_id, :role, :question_limit, :sender_email, :status, :created_at, :interview_scheduling_time)
            """)
        
        conn.execute(insert_invite, {
            "candidate_id": candidate_id,
            "role": role,
            "question_limit": question_limit,
            "sender_email": sender_email,
            "status": "Scheduled",
            "created_at": current_time,
            "interview_scheduling_time": current_time
        })

    st.success(f"Interview scheduled for '{name}' for role: {role}")

def track_candidate(name: Optional[str] = None, email: Optional[str] = None, role: Optional[str] = None, date_filter: Optional[str] = None) -> Union[list[dict], str]: 
    "Flexible candidate tracker. Filter by name, email, role, and date."
    try:
        query = """
            SELECT 
                c.id AS candidate_id,
                c.name AS name,
                c.email AS email,
                c.phone AS phone,

                t.role AS role,
                t.sender_email AS sender_email,
                t.status AS status,
                t.interview_scheduling_time AS interview_scheduling_time,

                d.achieved_score AS achieved_score,
                d.total_score AS total_score,
                d.summary AS summary,
                d.recommendation AS recommendation,
                d.skills AS skills

            FROM AI_INTERVIEW_PLATFORM.candidates c
            LEFT JOIN AI_INTERVIEW_PLATFORM.interview_invitation t ON c.id = t.candidate_id
            LEFT JOIN AI_INTERVIEW_PLATFORM.interview_details d ON t.id = d.candidate_id
            WHERE 1=1
        """
        params = {}

        if name:
            query += " AND LOWER(c.name) LIKE :name"
            params["name"] = f"%{name.strip().lower()}%"

        if email:
                query += " AND c.email = :email"
                params["email"] = email.strip().lower()

        if role:
            query += " AND LOWER(t.role) LIKE :role"
            params["role"] = f"%{role.lower()}%" 

        if date_filter:
            today = datetime.today()
            if date_filter == "last_week":
                start = today - timedelta(days=today.weekday() + 7)
                end = start + timedelta(days=6)
            elif date_filter == "recent":
                start = today - timedelta(days=3)
                end = today
            elif date_filter == "today":
                start = today.replace(hour=0, minute=0, second=0, microsecond=0)
                end = today
            else:
                start = None

            if start:
                query += " AND t.interview_scheduling_time BETWEEN :start AND :end"
                params["start"] = start
                params["end"] = end
        
        query += " ORDER BY c.created_at DESC"

        with engine.begin() as conn:
            result = conn.execute(text(query), params).mappings().all()

        if not result:
            return "No matching candidate records found."
        return [dict(row) for row in result]
    
    except Exception as e:
        return f"Error in tracking candidates: {str(e)}"
    
# To check available roles
def list_all_scheduled_roles() -> Union[list[str], str]:
    """Returns a list of all distinct roles for which interviews are scheduled."""
    try:
        query = """
            SELECT DISTINCT role from AI_INTERVIEW_PLATFORM.interview_invitation
            WHERE role is not NULL
            Order by role
        """
        with engine.begin() as conn:
            result = conn.execute(text(query)).scalars().all()

        if not result:
            return "No roles found with scheduled interviews."
        return result
    except Exception as e:
        return f"Error fetching roles: {str(e)}"

# Parsing part of Track Candidate
def extract_filters(user_input:str) -> dict:
    parsing_chain = parsing_prompt | llm | JsonOutputParser()
    parsing_result = parsing_chain.invoke({"input": user_input})

    return parsing_result


#Tools
    # To track candidate's details
# track_candidate_tool = StructuredTool.from_function(
#     func=track_candidate,
#     name='track_candidate',
#     description="Track candidates. Use email, name, role, and date filter to narrow down results.",
#     args_schema=TrackCandidateInput
# )

# Chatbot using Intent
# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# WhatsApp-like styling
st.markdown("""
<style>
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 160px);
        overflow-y: auto;
        padding: 10px;
        margin-bottom: 80px;
    }
    .user-message {
        align-self: flex-end;
        background-color: #DCF8C6;
        border-radius: 15px 15px 0 15px;
        padding: 10px 15px;
        margin: 5px;
        max-width: 70%;
    }
    .ai-message {
        align-self: flex-start;
        background-color: #ECECEC;
        border-radius: 15px 15px 15px 0;
        padding: 10px 15px;
        margin: 5px;
        max-width: 70%;
    }
    .input-container {
        position: fixed;
        bottom: 20px;
        left: 20px;
        right: 20px;
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def ask_ai():
    # Initialize components
    # memory = ConversationBufferMemory(k=20, memory_key="chat_history", return_messages=True)
    parser = StrOutputParser()
    rag_chain = create_rag_chain("formatted_QA.txt", prompt, parser)
    intent_chain = intent_prompt | llm | parser

    # tools = [track_candidate_tool]
    # agent = initialize_agent(
    #     tools=tools,
    #     llm=llm,
    #     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=False,
    #     memory=memory,
    #     handle_parsing_errors=True,
    #     max_iterations=3
    # )

    # Chat container
    chat_container = st.container()
    
    # Input container fixed at bottom
    with st.container():
        input_container = st.empty()
        with input_container.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input(
                    "Type your message...", 
                    label_visibility="collapsed",
                    placeholder="Type a message..."
                )
            with col2:
                submit = st.form_submit_button("Send", use_container_width=True)

    # Handle form submission
    if submit and user_input:
        try:
            intent = intent_chain.invoke(user_input)

            if intent == "bye":
                response = "Goodbye! Have a great day!"
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": response})
                if st.session_state.get("use_voice"):
                    st.markdown(speak_text(response), unsafe_allow_html=True)

            elif intent == "schedule_interview":
                # Store the intent to show form in next render
                st.session_state.current_intent = "schedule_interview"
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": "Let's schedule an interview. Please provide these details:"})

            elif intent == "list_roles":
                st.session_state.chat_history.append({"sender": "user", "content": user_input})

                # Call the role-fetching function
                response = list_all_scheduled_roles()

                # Now use the response
                if isinstance(response, str):
                    # It's an error message or "no roles found"
                    st.session_state.chat_history.append({"sender": "ai", "content": response})
                    if st.session_state.get("use_voice"):
                        st.markdown(speak_text(response), unsafe_allow_html=True)
                else:
                    # It's a list of roles
                    role_list_str = "\n".join(f"- {r}" for r in response)
                    summary = f"Interviews have been scheduled for the following roles:\n\n{role_list_str}"
                    st.session_state.chat_history.append({"sender": "ai", "content": summary})
                    if st.session_state.get("use_voice"):
                        st.markdown(speak_text("Scheduled interview roles include: " + ", ".join(response)), unsafe_allow_html=True)

            elif intent == "track_candidate":
                filters = extract_filters(user_input)
                response = track_candidate(
                    name=filters.get("name"),
                    email=filters.get("email"),
                    role=filters.get("role"),
                    date_filter=filters.get("date_filter")
                )

                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                if isinstance(response, str):
                    st.session_state.chat_history.append({"sender": "ai", "content": response})
                    if st.session_state.get("use_voice"):
                        st.markdown(speak_text(response), unsafe_allow_html=True)
                else:
                    # Add an AI message summary
                    st.session_state.chat_history.append({
                        "sender": "ai",
                        "content": f"Found {len(response)} matching candidate(s):"
                    })
                    if st.session_state.get("use_voice"):
                        st.markdown(speak_text(f"Found {len(response)} matching candidates"), unsafe_allow_html=True)

                    for row in response:
                        with st.container():
                            st.markdown("---")
                            st.markdown(f"### {row.get('name', 'NA')}  \n `{row.get('email', 'NA')}`  | `{row.get('phone', 'NA')}`")
                            st.markdown(f"**Role**: `{row.get('role', 'NA')}`  \n**Status**: `{row.get('status', 'NA')}`  \n**Scheduled At**: `{row.get('interview_scheduling_time', 'NA')}`")
                            st.markdown(f"**Achieved Score**: `{row.get('achieved_score', 'NA')}` / `{row.get('total_score', 'NA')}`")
                            st.markdown(f"**Recommendation**: `{row.get('recommendation', 'NA')}`  \n**Summary**: {row.get('summary', 'NA')}")
                            st.markdown(f"**Skills Evaluated**: `{row.get('skills', 'NA')}`")


            elif intent == "greet":
                response = llm.invoke(user_input).content
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": response})
                if st.session_state.get("use_voice"):
                    st.markdown(speak_text(response), unsafe_allow_html=True)

            elif intent == "help":
                response = rag_chain.invoke(user_input)
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": response})
                if st.session_state.get("use_voice"):
                    st.markdown(speak_text(response), unsafe_allow_html=True)

            else:
                response = "I'm sorry, I can only help with Jobma interview-related questions."
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": response})
                if st.session_state.get("use_voice"):
                    st.markdown(speak_text(response), unsafe_allow_html=True)

        except Exception as e:
            st.session_state.chat_history.append({"sender": "user", "content": user_input})
            st.session_state.chat_history.append({"sender": "ai", "content": f"Error: {str(e)}"})
            if st.session_state.get("use_voice"):
                st.markdown(speak_text(response), unsafe_allow_html=True)

    # Handle interview scheduling form
    if "current_intent" in st.session_state and st.session_state.current_intent == "schedule_interview":
        input_container.empty()  # Remove the input form temporarily
        
        with st.form("interview_form"):
            st.subheader("Interview Details")
            role = st.text_input("Target Role")
            uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
            question_limit = st.number_input("Number of Questions", min_value=1, value=5)
            sender_email = st.text_input("Your Email")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                submit_form = st.form_submit_button("Schedule Interview")
            with col2:
                cancel = st.form_submit_button("Cancel")
            
            if cancel:
                del st.session_state.current_intent
                st.rerun()
                
            if submit_form:
                if uploaded_resume and role and sender_email:
                    resume_path = os.path.join("temp_uploads", uploaded_resume.name)
                    os.makedirs("temp_uploads", exist_ok=True)
                    with open(resume_path, "wb") as f:
                        f.write(uploaded_resume.getbuffer())

                    try:
                        schedule_interview(
                            role=role,
                            resume_path=resume_path,
                            question_limit=question_limit,
                            sender_email=sender_email
                        )
                        st.session_state.chat_history.append({
                            "sender": "ai", 
                            "content": f"Interview scheduled successfully for '{role}'."
                        })
                        del st.session_state.current_intent
                        st.rerun()
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "sender": "ai",
                            "content": f"Error while scheduling: {e}"
                        })
                        st.rerun()
                else:
                    st.warning("Please fill all fields and upload a resume")

    # Display chat history
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history:
            if msg["sender"] == "user":
                st.markdown(
                    f'<div class="user-message">{msg["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="ai-message">{msg["content"]}</div>', 
                    unsafe_allow_html=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    ask_ai()