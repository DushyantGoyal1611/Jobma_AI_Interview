# This is the Interviewer's Side

import os
import re
import json
import warnings
from datetime import datetime
from dotenv import load_dotenv

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field  

# Code Starts from here --------------------------------------------->
warnings.filterwarnings('ignore')
load_dotenv()

# Input Schema Using Pydantic
class ScheduleInterviewInput(BaseModel):
    role:str = Field(description="Target Job Role")
    resume_path:str = Field(description="Path to resume file (PDF/DOCX/TXT)")
    question_limit:int = Field(description="Number of interview questions to generate")
    sender_email: str = Field(description="Sender's email address")

# For Current Day
current_month_year = datetime.now().strftime("%B %Y")

# Model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

# Parser
parser = StrOutputParser()

# Prompt
prompt = PromptTemplate(
    template="""
You are an intelligent assistant that only answers questions based on the provided document content.

The document may include:
- Headings, paragraphs, subheadings
- Lists or bullet points
- Tables or structured data
- Text from PDF, DOCX, or TXT formats

Your responsibilities:
1. Use ONLY the content in the document to answer.
2. If the question is clearly related to the document topic but the content is insufficient, respond with: INSUFFICIENT CONTEXT.
3. If the question is completely unrelated to the document, respond with: SORRY: This question is irrelevant.
4. If the user greets (e.g., "hi", "hello"), respond with a friendly greeting.
5. Otherwise, provide a concise and accurate answer using only the document content.

Document Content:
{context}

User Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# Intent Prompt
intent_prompt = PromptTemplate(
    template="""You are an AI Intent Classifier for the Jobma Interviewing Platform. Based on the user input, identify their intent from the list of predefined intents.

Possible Intents:
- **schedule_interview**: The user wants to schedule an interview, usually mentions a role, resume, job title, or similar context.
- **greet**: The user says hello, hi, good morning, or other greeting-like phrases.
- **help**: The user is asking for help or support about using the Jobma platform.
- **bye**: The user says goodbye or ends the conversation.
- **irrelevant**: The user input is unrelated to the Jobma platform or job interviews, such as asking about food, weather, sports, or general unrelated queries (e.g., "I want to make a pizza").

Classify the following user input strictly as one of the intents above. Your response must be a **single word** from the list: `schedule_interview`, `greet`, `help`, `bye`, or `irrelevant`.

User Input:
"{input}"

Intent:
""",
    input_variables=['input']
)

chain = intent_prompt | llm | parser

result = chain.invoke("What is 2+2")

print(result)