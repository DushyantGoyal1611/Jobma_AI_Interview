# This is the Candidate's Side (Updated)

import os
import re
import json
import warnings
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
# LangChain related libraries
# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Streamlit
st.set_page_config(page_title="Candidate Portal", layout="centered")
st.title("Candidate Interview Portal")

# Adding a Authentication Check
user_id = st.text_input("Enter your email ID").strip()
applied_role = st.text_input("Enter the role you applied for").strip()

# Extracting JSON
try:
    with open("interview_context.json", "r") as f:
        interview_context = json.load(f)
except FileNotFoundError:
    st.error("Interview context file not found.")
    st.stop()
except json.JSONDecodeError:
    st.error("Invalid JSON format in interview context file.")
    st.stop()

# Extracting Necessary Inputs from the JSON and Checking for matching Candidate
matched_data = None

# Only run matching if both fields are filled
if user_id and applied_role:
    for context in interview_context:
        if context.get('email', '').strip().lower() == user_id.lower() and context.get('target_role', '').strip().lower() == applied_role.lower():
            matched_data = context
            break

    if matched_data:
        st.success("Authentication successful. Proceeding...")
    else:
        st.warning("You are not scheduled for the interview. Please contact HR.")
        st.stop()
else:
    st.stop()

name = matched_data.get("name", "NA")
target_role = matched_data["target_role"]
skills = matched_data.get("skills", [])
experience = matched_data.get("experience", "NA")
email = matched_data["email"]
question_limit = matched_data.get("question_limit", 0)
sender_email = matched_data.get("sender_email", "NA")

# LLM to be used
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Initialize session state variables
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
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()


# Role-based Question-Generation
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
            if question not in st.session_state.asked_questions:
                st.session_state.asked_questions.add(question)
                return question
        except Exception as e:
            st.error(f"Error generating question: {e}")
    return "All unique questions have been exhausted."

# Chatbot
# Start button
if not st.session_state.interview_started:
    if st.button("Start Interview"):
        st.session_state.interview_started = True
        st.session_state.start_time = datetime.now()
        st.rerun()

# Interview loop
if st.session_state.interview_started:
    if st.session_state.counter < question_limit:
        if st.session_state.current_question == "":
            question = generate_next_question(experience, target_role)
            if question == "All unique questions have been exhausted.":
                st.write("AI: All unique questions have been exhausted.")
                st.session_state.interview_started = False
            else:
                st.session_state.current_question = question
                st.session_state.qna_history.append(AIMessage(content=question))

        if st.session_state.current_question:
            st.write(f"AI: {st.session_state.current_question}")
            interviewee_response = st.text_input("Your answer:", key=f"response_{st.session_state.counter}")

            if interviewee_response:
                st.session_state.qna_history.append(HumanMessage(content=interviewee_response))
                st.session_state.qa_pairs.append({
                    "Question": st.session_state.current_question,
                    "Answer": interviewee_response
                })
                st.session_state.counter += 1
                st.session_state.current_question = ""
                st.rerun()

    else:
        # End of interview
        end_time = datetime.now()
        time_taken = end_time - st.session_state.start_time
        st.session_state.time_taken = time_taken 
        st.write("Interview Ended. Thank you!")
        st.write(f"Total Questions Answered: {st.session_state.counter}")
        st.write(f"Time Taken: {time_taken}")

        # Optionally, show chat history
        if st.checkbox("Show Chat History"):
            st.subheader("Chat History:")
            for entry in st.session_state.qna_history:
                role = "AI" if isinstance(entry, AIMessage) else "You"
                st.write(f"**{role}:** {entry.content}")

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

# Extracts JSON safely from Gemini
def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]+\}', text)
        if json_match:
            return json.loads(json_match.group(0))
        raise ValueError("Could not extract valid JSON.")
    
if st.session_state.counter == question_limit:
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
        "time_taken": str(st.session_state.get("time_taken", "NA")),
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

    st.success(f"Candidate's report saved successfully to '{json_file}'")   

    st.info("Interview is complete. You may now close this screen. Thank you for your time and participation.")