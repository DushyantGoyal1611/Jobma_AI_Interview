import os
import re
import json
import warnings
from datetime import datetime
from dotenv import load_dotenv
# Libraries for Report Generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
# LangChain related libraries
# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# RAG Libraries
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# For Sending Mail
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# App Password (Used later for Mail Sending)
app_password = os.getenv("APP_PASSWORD")

# LLM to be used
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# For Current Day
current_month_year = datetime.now().strftime("%B %Y")

# Target Role
target_role = input("Target Role: ").strip()

if not target_role:
    print("Target Role not provided. Exiting .....")
    exit()

# Question Limit
while True:
    try:
        question_limit = int(input("How many Questions you want?: "))
        if question_limit > 0:
            break
        else:
            print("Please enter a number greater than 0.")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

# Output Parser
parser = JsonOutputParser()

# Skills and Experience fetching prompt
prompt = PromptTemplate(
    template="""
You are an AI Resume Analyzer. Analyze the resume text below and extract **only** information relevant to the given job role.

Your output **must** be in the following JSON format:
{format_instruction}

**Instructions:**

1. **Skills**:
   - Extract technical and soft skills relevant to the **target role**.
   - Exclude generic or irrelevant skills (e.g., MS Word, Internet Browsing).
   - If **no skills are relevant**, return an empty list: `"Skills": []`.

2. **Experience**:
   - Calculate the **cumulative time spent at each company** to get total professional experience.
   - Include only non-overlapping, clearly dated experiences (internships, jobs).
   - If a role ends in "Present" or "Current", treat it as ending in **{current_month_year}**.
   - Example: 
     - Google: Jan 2023 - Mar 2023 = 2 months  
     - Jobma: Feb 2025 - May 2025 = 3 months  
     - Total: 5 months = `"Experience": "0.42 years"`
   - Round the final answer to **2 decimal places**.
   - If durations are missing or unclear, return: `"Experience": "NA"`.

3. Fetch email id from the document
   - Extract the first valid email address ending with `@gmail.com` from the text.
   - If not found, return `"Email": "NA"`.

---

**Target Role**: {role}

**Resume Text**:
{context}
""",
    input_variables=["context", "role"],
    partial_variables={
        "format_instruction": parser.get_format_instructions(),
        "current_month_year": current_month_year
        }
)

def extract_document(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

        docs = loader.load_and_split()
        print(f'Loading {file_path} ......')
        print('Loading Successful')
        return docs
    except FileNotFoundError as fe:
        print(f'File not Found: {fe}')
    except Exception as e:
        print(f"Error loading document: {e}")

    return []

def rag_workflow(doc:str, prompt):
    # Document Loader
    docs = extract_document(doc)
    if not docs:
        raise ValueError("Error in Document Loading ....")
    
    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Embedding Model and Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs),
        "role": lambda _: target_role 
        }
        | prompt
        | llm
        | parser
    )

    return rag_chain

main_chain = rag_workflow("Ashutosh_AI_Engineer.pdf", prompt)

# Invoking
resume_result = main_chain.invoke("Extract skills, years of experience and mail id from this resume.")
print("Extracted: ",resume_result)

# Skills, Experience and Email Id
skills = resume_result.get("Skills", [])
experience = resume_result.get("Experience", "NA")
email = resume_result.get("Email", "NA")

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
    raise e

# Report Generation
def generate_pdf_report(feedback_json, time_taken, filename="Interview_Report.pdf", image_path=None):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 50
    margin = 50
    max_width = width - 2 * margin

    # Add image at the top-right
    if image_path:
        try:
            img_width = 80
            img_height = 80
            c.drawImage(image_path, width - margin - img_width, height - img_height - 20, width=img_width, height=img_height)
        except Exception as e:
            print(f"Error loading image: {e}")

    # Write the Title and Meta Info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Interview Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Email: {email}")
    y -= 20
    c.drawString(margin, y, f"Experience: {experience}")
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
        y = draw_wrapped_text(c, f"Score: {item['Score']}/10 | Feedback: {item['Comment']}", "Helvetica", 12, margin, y, max_width)
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
    y = draw_wrapped_text(c, feedback_json.get('Recommendation', 'No Recommendation from my side'), "Helvetica", 12, margin, y, max_width)

    c.save()

generate_pdf_report(feedback_data, time_taken, filename="test_report5.pdf", image_path="jobma_logo.png")
print("PDF Report Generated: Interview_Report.pdf")

@tool
def confirmation_mail():
    """This function will calculate score percentage and confirm if mail should be sent based on threshold."""
    threshold = 65
    feedback_items = feedback_data.get("Feedback", [])
    total_score = question_limit * 10
    achieved_score = 0

    for item in feedback_items:
        try:
            achieved_score += int(item.get("Achieved Score", 0))
        except Exception as e:
            continue
    
    if total_score == 0:
        return "Could not compute score percentage due to missing or invalid score data."
    
    score_percentage = (achieved_score / total_score) * 100

    print(f"Total Score: {total_score}")
    print(f"Achieved Score: {achieved_score}")
    print(f"Percentage: {score_percentage}")
    
    if score_percentage >= threshold:
        subject = "Interview Result: Passed"
        body = f"Dear {email}, \n\nCongratulations! You passed the AI Interview Round with the score of {score_percentage}% \nYou are proceeded to the next round. \n\nBest Regards,\nPehchan Kaun"
    else:
        return f"Candidate scored {score_percentage:.2f}%. Below threshold. No mail sent."
    
    # Prepare email
    msg = MIMEMultipart()
    sender_mail = str(input("Enter Sender's mail id: "))
    msg['From'] = sender_mail
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send mail via Gmail SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_mail, app_password)
        server.send_message(msg)
        server.quit()
        return f"Candidate passed with {score_percentage:.2f}% score. Confirmation mail sent to {email}."
    except Exception as e:
        return f"Score: {score_percentage:.2f}%. Failed to send email due to: {str(e)}"

result = confirmation_mail.invoke(input={})
print(result)