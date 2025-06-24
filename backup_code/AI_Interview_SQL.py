# This is the Candidate's Side (Updated)

import os
import re
import json
import warnings
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
    exit()

# Adding a Authentication Check
user_id = input("Enter your email id: ").strip()
applied_role = input("Enter the role you applied for: ").strip()

# SQL query to fetch candidate and scheduled interview
query = text("""
SELECT 
    c.id AS candidate_id,
    c.name,
    c.email,
    c.skills,
    c.experience,
    i.id AS invitation_id,
    i.role,
    i.question_limit,
    i.sender_email,
    i.status
FROM AI_INTERVIEW_PLATFORM.candidates c
JOIN AI_INTERVIEW_PLATFORM.interview_invitation i
    ON c.id = i.candidate_id
WHERE LOWER(c.email) = :email
  AND LOWER(i.role) = :role
  AND LOWER(i.status) = 'scheduled'
""")

with engine.connect() as conn:
    result = conn.execute(query, {"email": user_id, "role":applied_role}).fetchone()

if not result:
    print("You are not scheduled for the interview or invitation is not active. \nPlease Contact HR")
    exit()

candidate_id = result.candidate_id
invitation_id = result.invitation_id
name = result.name
target_role = result.role
experience = result.experience
email = result.email
question_limit = result.question_limit
sender_email = result.sender_email
if isinstance(result.skills, str) and result.skills.strip():
    try:
        skills = json.loads(result.skills)
    except json.JSONDecodeError:
        skills = [result.skills.strip()]  # fallback to treat as single skill string
else:
    skills = []  # empty list if no skills

# LLM to be used
@lru_cache(maxsize=1)
def get_llm():
    return ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0)

llm = get_llm()

# Chat History
qna_history = [
    SystemMessage(content="You are an AI Interviewer. Ask one question at a time based on candidate's resume. After the interviewee answers, ask the next one. Keep it conversational.")
]

# Role-based Question-Generation
asked_questions = set()
def generate_next_question(experience, target_role, max_retries=5):
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
            if question and question not in asked_questions:
                asked_questions.add(question)
                return question
        except Exception as e:
            print(f"Error generating question: {e}")
    return "All unique questions have been exhausted."

# Chatbot
print("Interview Started! \nType 'exit' to quit.")

counter = 0
qa_pairs = []
start_time = datetime.now()

while counter < question_limit:
    # Generate and print next question
    response = generate_next_question(experience, target_role)
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