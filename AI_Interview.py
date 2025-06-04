import os
import re
import json
import random
import warnings
from datetime import datetime
from dotenv import load_dotenv
# For User Interface
import streamlit as st
# Libraries for Report Generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader, simpleSplit
# LangChain related libraries
# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# RAG Libraries
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# LLM to be used
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

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
   - Calculate **total years of professional experience** based on time durations in the resume.
   - Example: 6 months + 1.5 years = 2.0 years.
   - If durations are missing or unclear, return: `"Experience": "NA"`.

---

**Target Role**: {role}

**Resume Text**:
{context}
""",
    input_variables=["context", "role"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
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

main_chain = rag_workflow("DushyantGoyal's Resume-hackerresume.pdf", prompt)

# Invoking
resume_result = main_chain.invoke("Extract skills and years of experience from this resume.")
print("Extracted: ",resume_result)

skills = resume_result.get("Skills", [])
experience = resume_result.get("Experience", "NA")
used_skills = set()

# Fail Safe 
if not skills:
    print(f"\nNo relevant skills found for the role '{target_role}'. Interview will not proceed.")
    exit() 

# Chat History
qna_history = [
    SystemMessage(content="You are an AI Interviewer. Ask one question at a time based on candidate's resume. After the interviewee answers, ask the next one. Keep it conversational.")
]

def generate_next_question(skills, experience, used_skills, target_role):
    remaining_skills = [skill for skill in skills if skill not in used_skills]

    if not remaining_skills:
        used_skills.clear()
        remaining_skills = skills
    selected_skill = random.choice(remaining_skills)
    used_skills.add(selected_skill)

    
    question_prompt = f"""
    You are an AI Interviewer interviewing a candidate for the role of "{target_role}".
    The candidate has {experience} years of experience and is skilled in: {', '.join(skills)}.

    Ask one interview question based on the skill: "{selected_skill}".

    Instructions:
    - The question should be relevant and appropriate to the level expected for a "{target_role}".
    - If it is a soft skill (e.g., Team Management, Communication, Leadership), ask a behavioral/situational question.
    - If it's a technical skill, ask a technical question.
    - Adjust the difficulty based on experience:
        - 0 to 2 years → basic to intermediate level question.
        - 2 to 6 years → intermediate to advanced level question.
        - More than 6 years → advanced level question.

    Only output the interview question. Do not include any explanations or extra text.
    """

    response = llm.invoke(question_prompt)
    return response, selected_skill

# Chatbot
print("Interview Started! \nType 'exit' to quit.")

counter = 0
qa_pairs = []
start_time = datetime.now()

while counter < question_limit:
# Generate and print next question
    response, skill_used = generate_next_question(skills, experience, used_skills, target_role)
    question = response.content.strip()
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
        "Answer": interviewee_response,
        "Skill": skill_used
    })
    counter += 1

end_time = datetime.now()
time_taken = end_time - start_time

print(f"Chat History:")
for chat in qna_history:
    print(chat.content)

# Feedback
feedback_prompt = f"""
    You are an AI Interview Assessor. Based on the following Q&A from an interview, provide:
    1. A score out of 10 for each answer.
    2. A brief feedback for each answer.
    3. An overall performance summary.

    Output in JSON format with:
    - "Feedback": List of {{"Question", "Answer", "Skill", "Score", "Comment"}}
    - "Summary": A short paragraph summarizing the candidate's performance.

    Q&A:
    {json.dumps(qa_pairs, indent=2)}
"""

# Extract JSON safely from Gemini
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

    for item in feedback_json['Feedback']:
        if y < 120:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12)

        c.setFont("Helvetica-Bold", 12)
        y = draw_wrapped_text(c, f"Skill: {item['Skill']}", "Helvetica-Bold", 12, margin, y, max_width)
        c.setFont("Helvetica", 12)
        y = draw_wrapped_text(c, f"Question: {item['Question']}", "Helvetica", 12, margin, y, max_width)
        y = draw_wrapped_text(c, f"Answer: {item['Answer']}", "Helvetica", 12, margin, y, max_width)
        y = draw_wrapped_text(c, f"Score: {item['Score']}/10 | Feedback: {item['Comment']}", "Helvetica", 12, margin, y, max_width)
        y -= 10

    c.setFont("Helvetica-Bold", 12)
    y = draw_wrapped_text(c, "Summary:", "Helvetica-Bold", 12, margin, y, max_width)
    c.setFont("Helvetica", 12)
    y = draw_wrapped_text(c, feedback_json['Summary'], "Helvetica", 12, margin, y, max_width)

    c.save()

generate_pdf_report(feedback_data, time_taken, filename="test_report2.pdf", image_path="jobma_logo.png")
print("PDF Report Generated: Interview_Report.pdf")