# This is the Candidate's Side (Updated)

import os
import re
import json
import warnings
from datetime import datetime
from dotenv import load_dotenv
# LangChain related libraries
# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# SQL
from sqlalchemy import create_engine
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

# Adding a Authentication Check
user_id = input("Enter your email id: ").strip()
applied_role = input("Enter the role you applied for: ").strip()

# Extracting JSON
with open("interview_context.json", "r") as f:
    interview_context = json.load(f)

# Extracting Necessary Inputs from the JSON and Checking for matching Candidate
matched_data = None
for context in interview_context:
    if context.get('email', '').strip().lower() == user_id.lower() and context.get('target_role', '').strip().lower() == applied_role.lower():
        matched_data = context
        break

# Handle if no match found
if not matched_data:
    print("You are not scheduled for the interview.\nPlease contact HR.")
    exit()

name = matched_data.get("name", "NA")
target_role = matched_data["target_role"]
skills = matched_data.get("skills", [])
experience = matched_data.get("experience", "NA")
email = matched_data["email"]
question_limit = matched_data.get("question_limit", 0)
sender_email = matched_data.get("sender_email", "NA")

# LLM to be used
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

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
            if question not in asked_questions:
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

result_context = {
    "name": name,
    "target_role": target_role,
    "skills": skills,
    "experience": experience,
    "email": email,
    "question_limit": question_limit,
    "time_taken": str(time_taken),
    "total_score":question_limit*10,
    "qna": qa_pairs,
    "achieved_score":achieved_score,
    "summary":summary,
    "recommendation":recommendation
}


json_file = "candidates_report.json"
data = []

if os.path.exists(json_file):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    except json.JSONDecodeError:
        data = []

data.append(result_context)

with open(json_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Candidate's Report Saved successfully and saved to '{json_file}'.")