# Candidate Side 20-06-2025, 14:14
# Using streamlit side by side (will test on frontend instead of CLI)
# Completed the Candidate Side's necessary changes 

import os
import re
import json
import warnings
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache
# LangChain related libraries
# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# SQL
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Streamlit
st.set_page_config(page_title="Candidate Portal", layout="centered")
st.title("Candidate Interview Portal")

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

engine = create_connection()
if not engine:
    print("Database connection failed.")
    st.stop()

# LLM
# @lru_cache(maxsize=1)
# def get_llm():
#     return ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

# llm = get_llm()
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

# Adding a Authentication Check
user_id = st.text_input("Enter your email ID")

# Will show all the Scheduled Interviews for Candidates
if user_id:
    user_id = user_id.strip().lower()
    with engine.begin() as conn:
        candidate_exists = conn.execute(text(
            """
                Select id from AI_INTERVIEW_PLATFORM.candidates where LOWER(email) = :email
            """
        ),
        {"email": user_id}
        ).scalar()

        if not candidate_exists:
            st.warning(f"No candidate found with email: {user_id}")
            st.stop()

        st.subheader("Your Scheduled Interviews:")

        scheduled_interviews = conn.execute(text(
            """
                Select c.id as candidate_id, t.id as invitation_id, c.skills, c.experience, t.question_limit, t.role from AI_INTERVIEW_PLATFORM.interview_invitation as t
                left join AI_INTERVIEW_PLATFORM.candidates as c 
                on c.id=t.candidate_id 
                where c.email = :email AND LOWER(t.status)='scheduled'
            """),
            {'email': user_id}
        ).mappings().all()

        if not scheduled_interviews:
            st.info(f"No Interview is Scheduled for {user_id} right now.")
        else:
            for idx, item in enumerate(scheduled_interviews):
                role = item.get('role', 'NA')
                experience = item.get('experience', 'NA')
                question_limit = item.get('question_limit', 0)
                invitation_id = item.get("invitation_id", 0)
                candidate_id = item.get("candidate_id", 0)
                skills_raw = item.get('skills', '')
                if isinstance(skills_raw, str) and skills_raw.strip():
                    try:
                        skills = json.loads(skills_raw)
                    except json.JSONDecodeError:
                        skills = [skills_raw.strip()]  # fallback to treat as single skill string
                else:
                    skills = []

                with st.container():
                    st.markdown("---")
                    if st.button(f"Start Interview for Role: {role}", key=f"start_role_{idx}"):
                        st.success(f"You’ve chosen to attend the interview for **{role}**.")
                        st.session_state.interview_started = True
                        st.session_state.role = role
                        st.session_state.experience = experience
                        st.session_state.question_limit = question_limit
                        st.session_state.invitation_id = invitation_id
                        st.session_state.candidate_id = candidate_id
                        st.session_state.skills = skills
                        st.session_state.counter = 0
                        st.session_state.qa_pairs = []
                        st.session_state.asked_questions = set()
                        st.session_state.qna_history = [
                            SystemMessage(content="You are an AI Interviewer. Ask one question at a time based on candidate's resume. After the interviewee answers, ask the next one. Keep it conversational.")
                        ]
                        st.session_state.start_time = datetime.now()
                        st.session_state.current_question = None
                        st.session_state.waiting_for_answer = False  
                        st.rerun()

else:
    st.info("Please enter your email to proceed.")


def initialize_session_state():
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    if "counter" not in st.session_state:
        st.session_state.counter = 0
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []
    if "asked_questions" not in st.session_state:
        st.session_state.asked_questions = set()
    if "qna_history" not in st.session_state:
        st.session_state.qna_history = [
            SystemMessage(content="You are an AI Interviewer. Ask one question at a time based on candidate's resume. After the interviewee answers, ask the next one. Keep it conversational.")
        ]
    if "skills" not in st.session_state:
        st.session_state.skills = []
    if "role" not in st.session_state:
        st.session_state.role = ""
    if "experience" not in st.session_state:
        st.session_state.experience = "0"
    if "question_limit" not in st.session_state:
        st.session_state.question_limit = 0
    if "invitation_id" not in st.session_state:
        st.session_state.invitation_id = 0
    if "candidate_id" not in st.session_state:
        st.session_state.candidate_id = 0
    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now()
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "end_time" not in st.session_state:
        st.session_state.end_time = None
    if "feedback_generated" not in st.session_state:
        st.session_state.feedback_generated = False
    if "interview_completed" not in st.session_state:
        st.session_state.interview_completed = False
    if "waiting_for_answer" not in st.session_state:
        st.session_state.waiting_for_answer = False 

initialize_session_state()

# Role-based Question-Generation
def generate_next_question(experience, target_role, max_retries=10):
    for _ in range(max_retries):
        question_prompt = f"""
            You are an AI Interviewer interviewing a candidate for the role of "{target_role}".
            The Candidate has {experience} years of experience.

            Generate one interview question that is:
            - Relevant and appropriate for the role: "{target_role}".
            - Calibrated to the candidate’s experience level:
                - 0 to 2 years → basic to intermediate level.
                - 2 to 6 years → intermediate to advanced level.
                - More than 6 years → advanced level.
            - You may include technical, situational, or behavioural elements as appropriate to the role.

            Only output the interview question. Do not include explanations or extra text.
        """

        try:
            response = llm.invoke(question_prompt)
            question = response.content.strip() if hasattr(response, "content") else str(response)
            if question and question not in st.session_state.asked_questions:
                st.session_state.asked_questions.add(question)
                return question
        except Exception as e:
            st.error(f"Error generating question: {e}")
            continue
    return "All unique questions have been exhausted."

# Extracts JSON safely from Gemini
def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]+\}', text)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError("Could not extract valid JSON.")

# Chatbot
if st.session_state.interview_started and not st.session_state.interview_completed:
    # Display progress
    st.write(f"Question {st.session_state.counter + 1} of {st.session_state.question_limit}")
    
    # Generate question if none is current
    if st.session_state.current_question is None or st.session_state.current_question in st.session_state.asked_questions:
        question = generate_next_question(st.session_state.experience, st.session_state.role)
        if question == "All unique questions have been exhausted.":
            st.error("No more unique questions available.")
            st.session_state.interview_completed = True
            st.session_state.end_time = datetime.now()
        else:
            st.session_state.current_question = question
            st.session_state.qna_history.append(AIMessage(content=question))
    
    # Display current question
    if st.session_state.current_question:
        st.write(f"AI: {st.session_state.current_question}")
        
        # Get candidate's response
        interviewee_response = st.text_input("Your answer:", key=f"response_{st.session_state.counter}")
        
        if interviewee_response:
            if interviewee_response.lower() == 'exit':
                st.session_state.interview_completed = True
                st.session_state.end_time = datetime.now()
                st.write("Interview ended by candidate.")
                st.rerun()
            else:
                # Store the Q&A pair
                st.session_state.qa_pairs.append({
                    "Question": st.session_state.current_question,
                    "Answer": interviewee_response
                })
                st.session_state.qna_history.append(HumanMessage(content=interviewee_response))
                
                # Move to next question
                st.session_state.counter += 1
                st.session_state.current_question = None
                
                # Check if interview is complete
                if st.session_state.counter >= st.session_state.question_limit:
                    st.session_state.interview_completed = True
                    st.session_state.end_time = datetime.now()
                
                # Force rerun to continue interview
                st.rerun()

# Show completion message if interview is done
if st.session_state.interview_completed:
    time_taken = st.session_state.end_time - st.session_state.start_time
    st.subheader("Interview Summary")
    st.write(f"Total Questions Answered: {st.session_state.counter}")
    st.write(f"Time Taken: {time_taken}")

if "end_time" not in st.session_state:
    st.session_state.end_time = None

if st.session_state.interview_completed and st.session_state.end_time:
    time_taken = st.session_state.end_time - st.session_state.start_time
    st.subheader("Interview Summary")
    st.write("Interview Ended. Thank you!")
    st.write(f"Total Questions Answered: {st.session_state.counter}")
    st.write(f"Time Taken: {time_taken}")

# ============================================================================

qa_pairs = st.session_state.qa_pairs
# Feedback
feedback_prompt = f"""
    You are an AI Interview Assessor. Based on the following Q&A from an interview, provide:
    1. A score out of 10 for each answer.
    2. A brief feedback for each answer.
    3. An overall performance summary.
    4. A final recommendation: "Recommended", "Borderline", or "Not Recommended".
    5. Score Candidate got.
    6. Total Score

    Output in JSON format with:
    - "Feedback": List of {{"Skill", "Question", "Answer", "Score", "Comment", "Achieved Score"}}
    - "Summary": A short paragraph summarizing the candidate's overall performance.
    - "Recommendation": One of "Recommended", "Borderline", or "Not Recommended".

    Q&A:
    {json.dumps(qa_pairs, indent=2)}
"""
    
feedback_response = llm.invoke(feedback_prompt)
try:
    feedback_data = extract_json(feedback_response.content)
except Exception as e:
    print("Invalid JSON from Gemini:\n", feedback_response.content)
    feedback_data = {
        "total_score": st.session_state.question_limit * 10,
        "achieved_score": 0,
        "summary": "Feedback could not be generated. Candidate may not have answered any questions."
    }

# Extract components like Summary and Recommendation
summary = feedback_data.get("Summary", "No Summary Provided")
recommendation = feedback_data.get("Recommendation", "No Recommendation Provided")

achieved_score = 0
for item in feedback_data.get("Feedback", []):
    try:
        achieved_score += int(item.get("Achieved Score", 0))
    except:
        continue

skills_str = json.dumps(st.session_state.skills, ensure_ascii=False)

# Separate questions and answers
questions_list = [pair["Question"] for pair in qa_pairs]
answers_list = [pair["Answer"] for pair in qa_pairs]

questions_json = json.dumps(questions_list, ensure_ascii=False)
answers_json = json.dumps(answers_list, ensure_ascii=False)
feedback_json = json.dumps(feedback_data.get("Feedback", []), ensure_ascii=False)

# Compose the SQL insert query
if st.session_state.interview_completed and st.session_state.end_time:
    insert_query = text("""
        INSERT INTO interview_details (
            invitation_id, candidate_id, questions, answers, achieved_score, total_score, feedback, summary, recommendation, skills
        ) VALUES (
            :invitation_id,
            :candidate_id,
            :questions,
            :answers,
            :achieved_score,
            :total_score,
            :feedback,
            :summary,
            :recommendation,
            :skills
        )
    """)

    try:
        with engine.begin() as conn:
            # Insert interview details
            conn.execute(insert_query, {
                "invitation_id": invitation_id,
                "candidate_id": candidate_id, 
                "questions": questions_json,
                "answers": answers_json,
                "achieved_score": achieved_score,
                "total_score": st.session_state.question_limit * 10,
                "feedback": feedback_json,
                "summary": summary,
                "recommendation": recommendation,
                "skills": skills_str
            })

            # Update interview status to COMPLETED
            update_status_query = text("""
                UPDATE AI_INTERVIEW_PLATFORM.interview_invitation
                SET status = :status
                WHERE id = :invitation_id
            """)

            conn.execute(update_status_query, {
                "status": "Completed",
                "invitation_id": invitation_id
            })

        st.success("Candidate's interview result saved to SQL database.")
        st.success(f"Interview status updated to 'Completed'")

    except Exception as e:
        st.error(f"Error saving interview result to SQL: {e}")

    st.info("Interview is complete. You may now close this screen. Thank you for your time and participation.")