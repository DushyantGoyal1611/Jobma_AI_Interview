import streamlit as st
from streamlit_chat import message
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, EmailStr

# Load environment variables
load_dotenv()

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'resume_path' not in st.session_state:
    st.session_state['resume_path'] = None

# Model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Input Schemas
class ScheduleInterviewInput(BaseModel):
    role: str = Field(description="Target Job Role")
    resume_path: str = Field(description="Path to resume file (PDF/DOCX/TXT)")
    question_limit: int = Field(description="Number of interview questions to generate")
    sender_email: str = Field(description="Sender's email address")

class TrackCandidateInput(BaseModel):
    email: EmailStr = Field(description="Email address of the Candidate")

# Function to Schedule Interview
def schedule_interview(role: str, resume_path: str, question_limit: int, sender_email: str) -> str:
    # Simplified for demo - you can keep your original implementation
    return f"Interview scheduled for {role} with {question_limit} questions. Confirmation sent to {sender_email}."

# Function to Track Candidate
def track_candidate(email: str) -> str:
    # Simplified for demo - you can keep your original implementation
    return f"Candidate with email {email} is in the interview process."

# Tools   
interview_tool = StructuredTool.from_function(
    func=schedule_interview,
    name='schedule_interview',
    description="Schedules an interview with role, resume path, question limit and sender email",
    args_schema=ScheduleInterviewInput
)   

track_candidate_tool = StructuredTool.from_function(
    func=track_candidate,
    name='track_candidate',
    description="Track candidates using their email address",
    args_schema=TrackCandidateInput
)

# Initialize agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [interview_tool, track_candidate_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    max_iterations=3
)

# Streamlit UI
st.set_page_config(page_title="Jobma Chat Assistant", page_icon="💬")

st.title("💬 Jobma Chat Assistant")
st.caption("A conversational AI assistant for interview scheduling and candidate tracking")

# Chat container
chat_container = st.container()

# Input form
with st.form(key='chat_form', clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Type your message:", 
            key='input',
            placeholder="Ask about scheduling interviews or tracking candidates...",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.form_submit_button(label='Send')

# Handle form submission
if submit_button and user_input:
    st.session_state['past'].append(user_input)
    
    with st.spinner("Thinking..."):
        try:
            # Get agent response
            response = agent.run(user_input)
            st.session_state['generated'].append(response)
        except Exception as e:
            st.session_state['generated'].append(f"Sorry, I encountered an error: {str(e)}")

# Display chat history
if st.session_state['generated']:
    with chat_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))

# File upload in sidebar
with st.sidebar:
    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader(
        "Select resume file (PDF/DOCX/TXT)", 
        type=['pdf', 'docx', 'txt'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Save the file
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state['resume_path'] = file_path
        st.success("Resume uploaded successfully!")