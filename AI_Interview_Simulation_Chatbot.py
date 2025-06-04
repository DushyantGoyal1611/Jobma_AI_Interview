import os
import re
import json
import random
import warnings
from datetime import datetime
from dotenv import load_dotenv

# Streamlit UI
import streamlit as st

# PDF Report
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader, simpleSplit

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

warnings.filterwarnings('ignore')
load_dotenv()

st.set_page_config(page_title="AI Interview Tool", layout="centered")
st.title("AI-Powered Interview Assistant")

# Sidebar Inputs
with st.sidebar:
    target_role = st.text_input("Target Role")
    question_limit = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    submit = st.button("Start Interview")

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
parser = JsonOutputParser()

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
   - If durations are missing or unclear, return: `"Experience": "NA"`.

**Target Role**: {role}

**Resume Text**:
{context}
""",
    input_variables=["context", "role"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

def extract_document(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload PDF or DOCX.")
    return loader.load_and_split()

def rag_workflow(doc_path, prompt):
    docs = extract_document(doc_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
    
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "role": lambda _: target_role}
        | prompt
        | llm
        | parser
    )
    return rag_chain

def generate_next_question(skills, experience, used_skills, target_role):
    remaining = [s for s in skills if s not in used_skills]
    if not remaining:
        used_skills.clear()
        remaining = skills
    skill = random.choice(remaining)
    used_skills.add(skill)

    q_prompt = f"""
You are an AI Interviewer interviewing a candidate for the role of "{target_role}".
The candidate has {experience} years of experience and is skilled in: {', '.join(skills)}.

Ask one interview question based on the skill: "{skill}".
Only output the interview question. No extra text.
"""
    response = llm.invoke(q_prompt)
    return response.content.strip(), skill

def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[\s\S]+\}', text)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Could not parse JSON.")

def generate_pdf_report(feedback_json, time_taken, filename="Interview_Report.pdf", image_path=None):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 50
    margin = 50
    max_width = width - 2 * margin

    if image_path:
        try:
            c.drawImage(image_path, width - margin - 80, height - 100, width=80, height=80)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Interview Report")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, f"Time Taken: {str(time_taken)}")
    y -= 30

    def draw_text(c, text, fontname, fontsize, x, y, max_width):
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
        c.setFont("Helvetica-Bold", 12)
        y = draw_text(c, f"Skill: {item['Skill']}", "Helvetica-Bold", 12, margin, y, max_width)
        c.setFont("Helvetica", 12)
        y = draw_text(c, f"Q: {item['Question']}", "Helvetica", 12, margin, y, max_width)
        y = draw_text(c, f"A: {item['Answer']}", "Helvetica", 12, margin, y, max_width)
        y = draw_text(c, f"Score: {item['Score']}/10 | Feedback: {item['Comment']}", "Helvetica", 12, margin, y, max_width)
        y -= 10

    y = draw_text(c, "Summary:", "Helvetica-Bold", 12, margin, y, max_width)
    y = draw_text(c, feedback_json['Summary'], "Helvetica", 12, margin, y, max_width)
    c.save()
    return filename

# Interview Logic
if submit and uploaded_file and target_role:
    with st.spinner("Processing resume..."):
        # Save uploaded file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        main_chain = rag_workflow(temp_path, prompt)
        result = main_chain.invoke("Extract skills and experience.")
        skills = result.get("Skills", [])
        experience = result.get("Experience", "NA")
        used_skills = set()

        os.remove(temp_path)

        if not skills:
            st.warning("No relevant skills found. Cannot proceed with the interview.")
        else:
            st.success(f"Resume analyzed: {experience} years experience | Skills: {', '.join(skills)}")
            qna_history = [SystemMessage(content="You are an AI Interviewer.")]
            qa_pairs = []
            start_time = datetime.now()
            count = 0

        if "count" not in st.session_state:
            st.session_state.count = 0
        if "qa_pairs" not in st.session_state:
            st.session_state.qa_pairs = []
        if "used_skills" not in st.session_state:
            st.session_state.used_skills = set()

        # Only run if interview is in progress
        if st.session_state.count < question_limit:
            question, skill = generate_next_question(skills, experience, st.session_state.used_skills, target_role)

            st.markdown(f"**Q{st.session_state.count + 1}: {question}**")

            answer_key = f"answer_{st.session_state.count}"
            submit_key = f"submit_{st.session_state.count}"

            answer = st.text_area("Your Answer", key=answer_key)

            if st.button("Submit", key=submit_key):
                st.session_state.qa_pairs.append({
                    "Question": question,
                    "Answer": answer,
                    "Skill": skill
                })
                st.session_state.count += 1
                st.experimental_rerun()

        # Evaluation triggers only after all questions are answered
        if len(st.session_state.qa_pairs) == question_limit:
            with st.spinner("Evaluating responses..."):
                feedback_prompt = f"""
        You are an AI Interview Assessor. Based on the following Q&A from an interview, provide:
        1. A score out of 10 for each answer.
        2. A brief feedback for each answer.
        3. An overall performance summary.

        Output in JSON format with:
        - "Feedback": List of {{ "Question", "Answer", "Skill", "Score", "Comment" }}
        - "Summary": A short paragraph summarizing the candidate's performance.

        Q&A:
        {json.dumps(st.session_state.qa_pairs, indent=2)}
        """
                feedback = llm.invoke(feedback_prompt)
                feedback_data = extract_json(feedback.content)
                end_time = datetime.now()
                time_taken = end_time - start_time

                report_path = generate_pdf_report(feedback_data, time_taken, "Interview_Report.pdf", "jobma_logo.png")

                st.success("Interview Complete!")
                st.markdown("### Summary:")
                st.write(feedback_data['Summary'])

                with open(report_path, "rb") as f:
                    st.download_button("Download PDF Report", f, file_name="Interview_Report.pdf", mime="application/pdf")
