# from langchain_core.tools import StructuredTool
# from langchain.agents import initialize_agent, AgentType
# from langchain.memory import ConversationBufferMemory
# from typing import Optional, Union, Literal
# from pydantic import BaseModel, Field, EmailStr
    


# def track_candidate(name: Optional[str] = None, email: Optional[str] = None, role: Optional[str] = None, date_filter: Optional[str] = None) -> Union[list[dict], str]: 
#     "Flexible candidate tracker. Filter by name, email, role, and date."
#     try:
#         query = """
#             SELECT 
#                 c.id AS candidate_id,
#                 c.name AS name,
#                 c.email AS email,
#                 c.phone AS phone,

#                 t.role AS role,
#                 t.sender_email AS sender_email,
#                 t.status AS status,
#                 t.interview_scheduling_time AS interview_scheduling_time,

#                 d.achieved_score AS achieved_score,
#                 d.total_score AS total_score,
#                 d.summary AS summary,
#                 d.recommendation AS recommendation,
#                 d.skills AS skills

#             FROM AI_INTERVIEW_PLATFORM.candidates c
#             LEFT JOIN AI_INTERVIEW_PLATFORM.interview_invitation t ON c.id = t.candidate_id
#             LEFT JOIN AI_INTERVIEW_PLATFORM.interview_details d ON t.id = d.candidate_id
#             WHERE 1=1
#         """
#         params = {}

#         if name:
#             query += " AND LOWER(c.name) LIKE :name"
#             params["name"] = f"%{name.strip().lower()}%"

#         if email:
#                 query += " AND c.email = :email"
#                 params["email"] = email.strip().lower()

#         if role:
#             query += " AND LOWER(t.role) LIKE :role"
#             params["role"] = f"%{role.lower()}%" 

#         if date_filter:
#             today = datetime.today()
#             if date_filter == "last_week":
#                 start = today - timedelta(days=today.weekday() + 7)
#                 end = start + timedelta(days=6)
#             elif date_filter == "recent":
#                 start = today - timedelta(days=3)
#                 end = today
#             elif date_filter == "today":
#                 start = today.replace(hour=0, minute=0, second=0, microsecond=0)
#                 end = today
#             else:
#                 start = None

#             if start:
#                 query += " AND t.interview_scheduling_time BETWEEN :start AND :end"
#                 params["start"] = start
#                 params["end"] = end
        
#         query += " ORDER BY c.created_at DESC"

#         with engine.begin() as conn:
#             result = conn.execute(text(query), params).mappings().all()

#         if not result:
#             return "No matching candidate records found."
#         return [dict(row) for row in result]
    
#     except Exception as e:
#         return f"Error in tracking candidates: {str(e)}"




# Memory
# memory = ConversationBufferMemory(k=20, memory_key="chat_history", return_messages=True)

# Tools
# tools = [track_candidate_tool]
# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=False,
#     memory=memory,
#     handle_parsing_errors=True,
#     max_iterations=3
# )




