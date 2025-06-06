import os
import re
import warnings
from dotenv import load_dotenv

# LangChain related libraries
# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# RAG Libraries
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.agents import Tool, initialize_agent, AgentType

# Memory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Code starts from here

warnings.filterwarnings('ignore')
load_dotenv()


# Tool
@tool
def calculation(a:int):
    """This function will solve any mathematical query"""
    return a**4

# Prompt
prompt = PromptTemplate(
    template="""
You are an intelligent assistant that answers user questions based only on the provided document content.
The things you have is document, some users may call it content or pages.

The documents (referred to as content, pages, or files) may include:
- Headings, subheadings, paragraphs
- Lists and bullet points
- Tables or structured data
- Text in PDF, DOCX, or TXT format

Instructions:
- If the question is a greeting (e.g., "hi", "hello"), respond with an appropriate greeting.
- If the question is a number, return its square using the calculation tool.
- If the question is irrelevant to the document, respond with: "SORRY: This question is irrelevant."
- If the context is insufficient to answer the question, respond with: INSUFFICIENT CONTEXT.
- Otherwise, answer as accurately and concisely as possible using only the provided context.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# Document Loader
def extract_document(file_path):
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")

        docs = loader.load()
        print(f'Loading {file_path} ......')
        print('Loading Successful')

        
        return docs

    except FileNotFoundError as fe:
        print(f"File not found: {fe}")
    except Exception as e:
        print(f"Error loading document: {e}")
    
    return []

# RAG WorkFlow
def rag_flow(doc, prompt):
    # Document Loader
    docs = extract_document(doc)
    if not docs:
        raise ValueError("Document could not be loaded.")
    # Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # Model Embeddings and Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_documents(chunks, embedding_model)
    # Retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_tool={'k':4})
    # Model
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    # Parser
    parser = StrOutputParser()

    def format_docs(retrieved_docs):
        return "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()    
    })

    main_chain = parallel_chain | prompt | llm | parser
    return main_chain


# The Chatbot
def ask_ai():   
    chat_history = [SystemMessage(content="You are a helpful AI Assistant")]
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    chain = rag_flow('formatted_QA.txt', prompt)

    agent = initialize_agent(
        tools= [calculation],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    while True:
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break

        if re.fullmatch(r"\d+", user_input):
            number = int(user_input)
            result = agent.run(f'Calculate the square of {number}')
            print(f"AI: {result}")
            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=str(result))
            ])
            continue

        response = chain.invoke(user_input)

        fallback_triggers = r"(insufficient|not (sure|enough|understand)|i don't know|no context)"
        if re.search(fallback_triggers, response, re.IGNORECASE):
            print("Fallback Triggered: Using Available AI Agent for external info... ")
            chat_history.append(HumanMessage(content=user_input))
            final_response = llm.invoke(chat_history)
            
            chat_history.append(final_response)
            print(f"AI (Fallback): {final_response.content}")
        else:
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            print(f"AI: {response}")

    print("Chat History:")
    print(chat_history)


# Calling the Chatbot
ask_ai()