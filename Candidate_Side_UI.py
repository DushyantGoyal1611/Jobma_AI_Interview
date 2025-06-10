# This is the Candidate's Side

import os
import re
import json
import warnings
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
# Libraries for Report Generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
# LangChain related libraries
# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
# RAG Libraries
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# For Sending Mail
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Streamlit
st.set_page_config(page_title="Candidate Portal", layout="centered")
st.title("Candidate Interview Portal")

# Adding a Authentication Check
user_id = st.text_input("Enter your email id: ")
applied_role = st.text_input("Enter the role you applied for: ")

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
for context in interview_context:
    if (context.get('email', '').strip().lower() == user_id.lower() and 
        context.get('target_role', '').strip().lower() == applied_role.lower()):
        matched_data = context
        break

# Handle if no match found
if not matched_data:
    st.warning("You are not scheduled for the interview. Please contact HR.")
    st.stop()
else:
    st.success("Authentication successful. Proceeding...")

name = matched_data.get("name", "NA")
target_role = matched_data.get("target_role", "NA")
skills = matched_data.get("skills", [])
experience = matched_data.get("experience", "NA")
email = matched_data.get("email", "NA")
question_limit = matched_data.get("question_limit", 0)
sender_email = matched_data.get("sender_email", "NA")

# App Password (Used later for Mail Sending)
app_password = os.getenv("APP_PASSWORD")
if not app_password:
    st.warning("APP_PASSWORD environment variable not set. Email functionality will be disabled.")

# LLM to be used
try:
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
except Exception as e:
    st.error(f"Failed to initialize LLM: {e}")
    st.stop()

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
            - Calibrated to the candidate's experience level:
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
        st.write("Interview Ended. Thank you!")
        st.write(f"Total Questions Answered: {st.session_state.counter}")
        st.write(f"Time Taken: {time_taken}")
        
        # Optionally, show chat history
        if st.checkbox("Show Chat History"):
            st.subheader("Chat History:")
            for entry in st.session_state.qna_history:
                role = "AI" if isinstance(entry, AIMessage) else "You"
                st.write(f"**{role}:** {entry.content}")

        # Generate feedback
        if st.button("Generate Feedback"):
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
                - "TotalScore": The total possible score.
                - "AchievedScore": The score the candidate achieved.

                Q&A:
                {json.dumps(st.session_state.qa_pairs, indent=2)}
            """

            # Extracts JSON safely from Gemini
            def extract_json(text):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{[\s\S]+\}', text)
                    if json_match:
                        try:
                            return json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            pass
                    st.error("Could not extract valid feedback from the AI.")
                    return None

            try:
                feedback_response = llm.invoke(feedback_prompt)
                feedback_data = extract_json(feedback_response.content)
                
                if feedback_data:
                    st.session_state.feedback_data = feedback_data
                    st.success("Feedback generated successfully!")
            except Exception as e:
                st.error(f"Error generating feedback: {e}")

        # Display feedback if available
        if "feedback_data" in st.session_state:
            st.subheader("Interview Feedback")
            
            # Display individual feedback
            for item in st.session_state.feedback_data.get("Feedback", []):
                with st.expander(f"Question: {item.get('Question', '')}"):
                    st.write(f"**Your Answer:** {item.get('Answer', '')}")
                    st.write(f"**Score:** {item.get('Score', '')}/10")
                    st.write(f"**Feedback:** {item.get('Comment', '')}")
            
            # Display summary
            st.subheader("Overall Summary")
            st.write(st.session_state.feedback_data.get("Summary", "No summary available."))
            
            # Display recommendation
            recommendation = st.session_state.feedback_data.get("Recommendation", "No recommendation")
            st.subheader("Recommendation")
            st.write(recommendation)
            
            # Generate PDF report
            if st.button("Generate PDF Report"):
                def generate_pdf_report(feedback_json, time_taken, filename="Interview_Report.pdf", image_path=None):
                    c = canvas.Canvas(filename, pagesize=A4)
                    width, height = A4
                    y = height - 50
                    margin = 50
                    max_width = width - 2 * margin

                    # Add image at the top-right if available
                    if image_path and os.path.exists(image_path):
                        try:
                            img_width = 80
                            img_height = 80
                            c.drawImage(image_path, width - margin - img_width, height - img_height - 20, 
                                       width=img_width, height=img_height)
                        except Exception as e:
                            print(f"Error loading image: {e}")

                    # Write the Title and Meta Info
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(margin, y, "Interview Report")
                    y -= 30

                    c.setFont("Helvetica", 12)
                    c.drawString(margin, y, f"Name: {name}")
                    y -= 20
                    c.drawString(margin, y, f"Email: {email}")
                    y -= 20
                    c.drawString(margin, y, f"Experience: {experience} years")
                    y -= 20
                    c.drawString(margin, y, f"Skills: {', '.join(skills)}")
                    y -= 20
                    c.drawString(margin, y, f"Time Taken: {str(time_taken)}")
                    y -= 30
                    c.drawString(margin, y, "Candidate Performance:")
                    y -= 20

                    def draw_wrapped_text(c, text, fontname, fontsize, x, y, max_width):
                        lines = simpleSplit(text, fontname, fontsize, max_width)
                        for line in lines:
                            if y < 100:
                                c.showPage()
                                y = height - 50
                                c.setFont(fontname, fontsize)
                            c.drawString(x, y, line.strip())
                            y -= 16
                        return y

                    for item in feedback_json.get('Feedback', []):
                        if y < 120:
                            c.showPage()
                            y = height - 50
                            c.setFont("Helvetica", 12)

                        if 'Skill' in item:
                            y = draw_wrapped_text(c, f"Skill: {item['Skill']}", "Helvetica-Bold", 12, margin, y, max_width)
                        y = draw_wrapped_text(c, f"Question: {item['Question']}", "Helvetica", 12, margin, y, max_width)
                        y = draw_wrapped_text(c, f"Answer: {item['Answer']}", "Helvetica", 12, margin, y, max_width)
                        y = draw_wrapped_text(c, f"Score: {item.get('Score', 'N/A')}/10 | Feedback: {item.get('Comment', 'No feedback')}", 
                                              "Helvetica", 12, margin, y, max_width)
                        y -= 10

                    # Summary
                    c.setFont("Helvetica-Bold", 12)
                    y = draw_wrapped_text(c, "Summary:", "Helvetica-Bold", 12, margin, y, max_width)
                    c.setFont("Helvetica", 12)
                    y = draw_wrapped_text(c, feedback_json.get('Summary', 'No Summary Provided'), "Helvetica", 12, margin, y, max_width)

                    # Recommendation
                    c.setFont("Helvetica-Bold", 12)
                    y = draw_wrapped_text(c, "Recommendation:", "Helvetica-Bold", 12, margin, y, max_width)
                    c.setFont("Helvetica", 12)
                    y = draw_wrapped_text(c, feedback_json.get('Recommendation', 'No Recommendation provided'), "Helvetica", 12, margin, y, max_width)

                    # Scores
                    c.setFont("Helvetica-Bold", 12)
                    y = draw_wrapped_text(c, "Scores:", "Helvetica-Bold", 12, margin, y, max_width)
                    c.setFont("Helvetica", 12)
                    y = draw_wrapped_text(c, f"Achieved Score: {feedback_json.get('AchievedScore', 'N/A')} out of {feedback_json.get('TotalScore', 'N/A')}", 
                                          "Helvetica", 12, margin, y, max_width)

                    c.save()

                try:
                    generate_pdf_report(
                        st.session_state.feedback_data, 
                        time_taken, 
                        filename=f"Interview_Report_{name.replace(' ', '_')}.pdf",
                        image_path="jobma_logo.png" if os.path.exists("jobma_logo.png") else None
                    )
                    st.success("PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Failed to generate PDF report: {e}")

                # Send confirmation email if applicable
                # Send confirmation email if applicable
if app_password and sender_email and email:
    def send_confirmation_email():
        """Send confirmation email based on candidate's performance."""
        threshold = 65
        feedback_data = st.session_state.feedback_data
        
        try:
            total_score = int(feedback_data.get("TotalScore", 0))
            achieved_score = int(feedback_data.get("AchievedScore", 0))
        except (ValueError, TypeError):
            return "Could not compute score percentage due to missing or invalid score data."
        
        if total_score == 0:
            return "Could not compute score percentage due to zero total score."
        
        score_percentage = (achieved_score / total_score) * 100

        if score_percentage >= threshold:
            subject = "Interview Result: Passed"
            body = f"""Dear {name}, 

            Congratulations! You passed the AI Interview Round with a score of {score_percentage:.2f}%. 
            You will proceed to the next round. 

            Best Regards,
            Pehchan Kaun"""
        else:
            return f"Candidate scored {score_percentage:.2f}%. Below threshold. No mail sent."
        
        # Prepare email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Send mail via Gmail SMTP server
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, app_password)
                server.send_message(msg)
            return f"Candidate passed with {score_percentage:.2f}% score. Confirmation mail sent to {email}."
        except Exception as e:
            return f"Score: {score_percentage:.2f}%. Failed to send email due to: {str(e)}"

    if st.button("Send Result Email"):
        result = send_confirmation_email()
        st.write(result)