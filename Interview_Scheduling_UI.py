# This is the Interviewer's Side (Updated)

import os
import json
import warnings
import streamlit as st
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
from langchain.memory import ConversationBufferMemory
from typing import Optional
from pydantic import BaseModel, Field, EmailStr

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Streamlit for User Interface
st.set_page_config(page_title="AI Interview Assistant", layout="centered")
st.title("AI Interview Assistant")

# Input Schema Using Pydantic
    # For Interview Scheduling
class ScheduleInterviewInput(BaseModel):
    role:str = Field(description="Target Job Role")
    resume_path:str = Field(description="Path to resume file (PDF/DOCX/TXT)")
    question_limit:int = Field(description="Number of interview questions to generate")
    sender_email:str = Field(description="Sender's email address")

    # For Tracking Candidate
class TrackCandidateInput(BaseModel):
    email:Optional[EmailStr] = None

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
- **bye**: The user says goodbye or ends the conversation.
- **irrelevant**: The user input is unrelated to the Jobma platform or job interviews, such as asking about food, weather, sports, or general unrelated queries (e.g., "I want to make a pizza").

Classify the following user input strictly as one of the intents above. Your response must be a **single word** from the list: `schedule_interview`, `greet`, `help`, `bye`, or `irrelevant`.

User Input:
"{input}"

Intent:
""",
    input_variables=['input']
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
        st.write(f'Loading {file_path} ......')
        st.write('Loading Successful')
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
        st.error("resume_path must be a valid string")

    if not os.path.isfile(resume_path):
        st.error(f"Resume file not found at path: {resume_path}")

    if isinstance(role, dict):
        role = role.get("title", "")
        if not role:
            role = str(role) 
    
    if not isinstance(role, str) or not role.strip():
        st.error("Role must be a non-empty string")

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

# Function to Track Candidate's Details using their email id
def track_candidate(email:str="") -> dict | list | str:
    """ Track candidate(s) by email. If no email is provided, return all candidate records. """
    try:
        with open("candidates_report.json", "r") as f:
            candidates = json.load(f)

        if not isinstance(candidates, list):
            candidates = [candidates]

        if not candidates:
            return "No candidates found in the database."

        if email:
            for candidate in candidates:
                if candidate.get('email', '').lower() == email.lower():
                    return json.dumps(candidate, indent=2)
            return f"No candidate found with email {email}."
        
        return json.dumps(candidates, indent=2)
        
    except Exception as e:
        return f"Error occurred: {str(e)}"


# Tools   
interview_tool = StructuredTool.from_function(
    func=schedule_interview,
    name='schedule_interview',
    description="Extracts resume information and schedules interview. Input should be a dictionary with keys: role, resume_path, question_limit, sender_email",
    args_schema=ScheduleInterviewInput
)   

track_candidate_tool = StructuredTool.from_function(
    func=track_candidate,
    name='track_candidate',
    description="Track candidates. Provide email to get specific candidate details or leave blank to get a summary of all interviews.",
    args_schema=TrackCandidateInput
)

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
        height: calc(100vh - 200px);
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
    st.title("Jobma Interview Assistant")

    # Initialize components
    memory = ConversationBufferMemory(k=20, memory_key="chat_history", return_messages=True)
    parser = StrOutputParser()
    rag_chain = create_rag_chain("formatted_QA.txt", prompt, parser)
    intent_chain = intent_prompt | llm | parser

    tools = [interview_tool, track_candidate_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=3
    )

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

            elif intent == "schedule_interview":
                # Store the intent to show form in next render
                st.session_state.current_intent = "schedule_interview"
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": "Let's schedule an interview. Please provide these details:"})

            elif intent == "track_candidate":
                agent_response = agent.invoke({'input': user_input})
                response = agent_response['output']
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": response})

            elif intent == "greet":
                response = llm.invoke(user_input).content
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": response})

            elif intent == "help":
                response = rag_chain.invoke(user_input)
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": response})

            else:
                response = "I'm sorry, I can only help with Jobma interview-related questions."
                st.session_state.chat_history.append({"sender": "user", "content": user_input})
                st.session_state.chat_history.append({"sender": "ai", "content": response})

        except Exception as e:
            st.session_state.chat_history.append({"sender": "user", "content": user_input})
            st.session_state.chat_history.append({"sender": "ai", "content": f"Error: {str(e)}"})

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
                        response = agent.run(
                            f"Schedule an interview for {role} using resume at {resume_path}. "
                            f"Generate {question_limit} questions and send confirmation to {sender_email}."
                        )
                        st.session_state.chat_history.append({"sender": "ai", "content": response})
                        del st.session_state.current_intent
                        st.rerun()
                    except Exception as e:
                        st.session_state.chat_history.append({"sender": "ai", "content": f"Error: {str(e)}"})
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