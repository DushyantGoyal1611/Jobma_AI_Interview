# Previously used Document related Prompt
# prompt = PromptTemplate(
#     template="""
# You are an intelligent assistant that only answers questions based on the provided document content.

# The document may include:
# - Headings, paragraphs, subheadings
# - Lists or bullet points
# - Tables or structured data
# - Text from PDF, DOCX, or TXT formats

# Your responsibilities:
# 1. Use ONLY the content in the document to answer.
# 2. If the question is clearly related to the document topic but the content is insufficient, respond with: INSUFFICIENT CONTEXT.
# 3. If the question is completely unrelated to the document, respond with: SORRY: This question is irrelevant.
# 4. If the user greets (e.g., "hi", "hello"), respond with a friendly greeting.
# 5. Otherwise, provide a concise and accurate answer using only the document content.

# Document Content:
# {context}

# User Question:
# {question}

# Answer:
# """,
#     input_variables=["context", "question"]
# )


# Built-in Memory used in this Chatbot
# def ask_ai():
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#     parser = StrOutputParser()
#     rag_chain = create_rag_chain("formatted_QA.txt", prompt, parser)

#     tools = [interview_tool]
#     agent = initialize_agent(
#         tools=tools,
#         llm=llm,
#         agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#         memory=memory,
#         verbose=True
#     )

#     fallback_triggers = re.compile(r"(insufficient|not (sure|enough|understand)|i don't know|no context)", re.IGNORECASE)

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ['exit', 'quit']:
#             print("Exiting Chat.\nGoodbye!")
#             break

#         intent = detect_intent(user_input)
        
#         # Agent Call for Schedule Interview
#         if user_input.lower().startswith("schedule interview"):
#             print("Invoking Interview Scheduler Agent...")
#             try:
#                 role = input("Enter Target Role: ")
#                 resume_path = input("Enter Resume Path: ")
#                 if not os.path.isfile(resume_path):
#                     print(f"Error: File not found at {resume_path}")
#                     continue
#                 question_limit = int(input("Question Limit: "))
#                 sender_email = input("Sender's Email: ")

#                 tool_input = {
#                     "role": role,
#                     "resume_path": resume_path,
#                     "question_limit": question_limit,
#                     "sender_email": sender_email
#                 }
#                 print("Agent Response:", agent.invoke({"input": f"schedule interview", **tool_input}))
#                 continue

#             except Exception as e:
#                 print("Error during scheduling:", str(e))
#                 continue

#         # RAG Response
#         response = rag_chain.invoke(user_input)
#         if fallback_triggers.search(response):
#             print("Fallback Triggered. Using LLM for external info...")
#             fallback_response = agent.invoke({"input": user_input})
#             print(f"AI (Fallback): {fallback_response['output']}")
#         else:
#             print(f"AI: {response}")

# ask_ai()


# Calling agent based on intent
#    elif response == 'schedule_interview':
#                 agent_response = agent.invoke({'input': user_input})
#                 print(f"Agent({response}): {agent_response['output']}")
#                 continue