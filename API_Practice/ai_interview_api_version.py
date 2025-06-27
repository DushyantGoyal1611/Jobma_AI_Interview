# Candidate Side
# This file is created just for practicing building APIs
# Modified on 27-06-2025 at 12:08

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

# SQL Connection
@lru_cache(maxsize=1)
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
    exit()

# LLM
@lru_cache(maxsize=1)
def get_llm():
    try:
        return ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            temperature=0.5,
            max_retries=3,
            request_timeout=30
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return None

llm = get_llm()
if not llm:
    print("Critical error: Could not initialize LLM. Please try again later.")
    exit()

# Adding a Authentication Check
user_id = input("Enter your email id: ").strip()

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
            print(f"No Candidate found with email: {user_id}")
            
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
            print(f"No Interview is Scheduled for {user_id} right now")
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

                print(f"{role}")

# Adding a Authentication Check for Scheduled Interview
selected_role = input("Enter the role you want to proceed with: ").strip()

# Role-based Question-Generation
asked_questions = set()
def generate_next_question(experience, target_role, max_retries=3):
    question_prompt = f"""
        You are an AI Interviewer conducting a structured interview for the role of "{target_role}".

        Candidate Profile:
        - Experience: {experience} years

        Instructions:
        - Generate one unique and relevant interview question tailored to the candidate’s experience and the role.
        - Use a mix of technical, behavioral, and situational styles if appropriate.
        - Do not repeat previous questions or themes.
        - Make sure the question is meaningful and precise.

        Only output the interview question. No explanations or extra text.
    """
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(question_prompt)
            question = response.content.strip() if hasattr(response, "content") else str(response)
            print(f"[Retry {attempt+1}] Question: {question}")
            
            question = response.content.strip() if hasattr(response, "content") else str(response)
            if question and question not in asked_questions:
                asked_questions.add(question)
                return question
        except Exception as e:
            print(f"Error generating question: {e}")
            continue
    return "All unique questions have been exhausted."

counter = 0
qa_pairs = []
start_time = datetime.now()

# Chat History
qna_history = [
    SystemMessage(content="You are an AI Interviewer. Ask one question at a time based on candidate's resume. After the interviewee answers, ask the next one. Keep it conversational.")
]

# Chatbot
print("Interview Started! \nType 'exit' to quit.")

counter = 0
qa_pairs = []
start_time = datetime.now()

while counter < question_limit:
    # Generate and print next question
    response = generate_next_question(experience, selected_role)
    if response == "All unique questions have been exhausted.":
        print("AI: All unique questions have been exhausted.")
        break
    question = response.strip()
    print(f"AI: {question}")

    qna_history.append(AIMessage(content=question))

    # Wait for candidate's response
    interviewee_response = input("You: ")
    if interviewee_response.lower() == 'exit':
        print("Interview Ended. \nThank you!")
        break
    
    qna_history.append(HumanMessage(content=interviewee_response))
    qa_pairs.append({
        "Question": question,
        "Answer": interviewee_response
    })
    counter += 1

end_time = datetime.now()
time_taken = end_time - start_time

# To see the Chat History
print(f"Chat History:")
for chat in qna_history:
    print(chat.content)

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

# Extracts JSON safely from Gemini
def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]+\}', text)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError("Could not extract valid JSON.")

feedback_response = llm.invoke(feedback_prompt)
try:
    feedback_data = extract_json(feedback_response.content)
except Exception as e:
    print("Invalid JSON from Gemini:\n", feedback_response.content)
    feedback_data = {
        "total_score": question_limit * 10,
        "achieved_score": 0,
        "summary": "Feedback could not be generated. Candidate may not have answered any questions."
    }

# Extract components
summary = feedback_data.get("Summary", "No Summary Provided")
recommendation = feedback_data.get("Recommendation", "No Recommendation Provided")

achieved_score = 0
for item in feedback_data.get("Feedback", []):
    try:
        achieved_score += int(item.get("Achieved Score", 0))
    except:
        continue

skills_str = json.dumps(skills, ensure_ascii=False)

# Separate questions and answers
questions_list = [pair["Question"] for pair in qa_pairs]
answers_list = [pair["Answer"] for pair in qa_pairs]

questions_json = json.dumps(questions_list, ensure_ascii=False)
answers_json = json.dumps(answers_list, ensure_ascii=False)
feedback_json = json.dumps(feedback_data.get("Feedback", []), ensure_ascii=False)

# Compose the SQL insert query
insert_query = text("""
    INSERT INTO interview_details (
        invitation_id,
        candidate_id,
        questions,
        answers,
        achieved_score,
        total_score,
        feedback,
        summary,
        recommendation,
        skills
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
            "total_score": question_limit * 10,
            "feedback": feedback_json,
            "summary": summary,
            "recommendation": recommendation,
            "skills": skills_str
        })

        # Update interview status to COMPLETED (move this here)
        update_status_query = text("""
            UPDATE AI_INTERVIEW_PLATFORM.interview_invitation
            SET status = :status
            WHERE id = :invitation_id
        """)

        conn.execute(update_status_query, {
            "status": "Completed",
            "invitation_id": invitation_id
        })

    print("Candidate's interview result saved to SQL database.")
    print(f"Interview status updated to 'Completed'")

except Exception as e:
    print("Error saving interview result to SQL:", e)