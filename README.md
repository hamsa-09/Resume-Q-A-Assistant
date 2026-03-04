import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="""
You are a weather assistant.

When the get_weather tool is called,
you MUST return exactly the tool output.
Do not add commentary.
Do not contradict the tool.
Do not mention real-time access.
"""
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response["messages"][-1].content)


------------------------------------------------
python main.py

python -m pip install langchain langchain-groq langchain-community faiss-cpu python-dotenv pypdf sentence-transformers
--------------------------------------------------
# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.tools import tool
# from langchain.agents import create_agent

# load_dotenv()

# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="openai/gpt-oss-120b",
# )

# @tool
# def get_weather(city: str) -> str:
#     """
#     Get weather for a given city.
#     Use this tool whenever the user asks about weather.
#     """
#     return f"It's always sunny in {city}!"

# agent = create_agent(
#     model=llm,
#     tools=[get_weather],
#     system_prompt="""
# You are a weather assistant.
# Always use the get_weather tool for weather-related questions.
# Return exactly the tool output.
# Do not add extra commentary.
# """
# )

# messages = []

# while True:
#     user_input = input("\nAsk: ")
#     if user_input.lower() == "exit":
#         break

#     messages.append({"role": "user", "content": user_input})

#     response = agent.invoke({"messages": messages})

#     final_message = response["messages"][-1]

#     print("\nAI:", final_message.content)

#     messages.append({"role": "assistant", "content": final_message.content})

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
)

# Inject Resume Into System Prompt
resume_context = """
You are a resume assistant for Hamsavardhini B.

You must answer ONLY using the information below.
If the answer is not found in the resume, say exactly:

"I don't have that information in my resume."

-------------------------
RESUME:


Hamsavardhini B
Email: hamsavardhinibaskar@gmail.com                    | Mobile: +91 9677838659
LinkedIn: www.linkedin.com/in/hamsavardhinibaskar  |  GitHub: github.com/hamsa-09
Leetcode: leetcode.com/u/Hamsa_09                          | GeeksForGeeks: www.geeksforgeeks.org/profile/hamsavardhini
ABOUT ME
      Full Stack Developer with hands-on experience building scalable web applications and AI-powered solutions. Skilled in developing responsive frontend interfaces and secure backend systems. Strong foundation in problem-solving and full-stack development. Passionate about building efficient, user-focused, and production-ready software solutions.

EDUCATION
Bannari Amman Institute of Technology, Erode	       Erode, India
Bachelor of Engineering in Computer Science and Engineering | CGPA: 8.3		   June 2022 - June 2026

SKILLS SUMMARY
Languages  : Java, JavaScript, TypeScript, Python
Frameworks: Spring Boot, FastAPI, React.js, Angular, Express.js, LangChain, LangGraph
Databases   : PostgreSQL, MySQL, MongoDB, Prisma ORM
Tools            : Git, GitHub, Docker, Postman, Podman

WORK EXPERIENCE
Software Development Intern | HashedIn By Deloitte | Jan 2026 – Present
Project: AI Product-to-Code – Multi-Agent Implementation & Validation System (GenAI)
Built a multi-agent AI system using Python and FastAPI to transform product requirements into structured implementation outputs.
 Designed AI agents for planning, task decomposition, code generation, and automated validation.
 Implemented RAG with embeddings and orchestrated workflows using LangGraph with HITL approval checkpoints.
 Developed a spec-driven validation pipeline for acceptance criteria, dependencies, and edge cases.
Project: VertexSpace – Workspace Booking System (Full Stack)
Engineered a full-stack workplace resource booking platform using React.js, Tailwind CSS, Spring Boot, and PostgreSQL as part of an enterprise internship project.
Designed and implemented secure REST APIs for managing rooms, desks, and parking reservations with role-based access control.
Applied concurrency-safe booking workflows using transactional control to prevent double bookings and ensure data consistency.
Integrated WebSockets to enable real-time booking status updates and live availability tracking.
Developed scheduling logic with lifecycle-based state transitions and configurable post-booking buffer rules.
Standardized UTC-based time storage with business-rule-driven timezone conversion for consistent cross-region scheduling.

PROJECTS
HomeSolutions – Home Services Platform (Backend)	                            			         	                            Feb 26
Developed a secure and scalable RESTful backend for an on-demand home services platform using Spring Boot and PostgreSQL.
 Implemented JWT-based authentication and role-based access control for Customer, Expert, and Admin workflows.
Built end-to-end booking lifecycle management with controlled state transitions from payment to service completion.
Developed service discovery APIs with pagination, filtering, and sorting for efficient catalog browsing.
Added expert onboarding, admin approval flows, and role-restricted job lifecycle operations.
Transport Permit Management System (Full Stack)						         		               Oct 25
Developed a full-stack transport permit management system to digitize application and approval workflows.
Built secure authentication and role-based access control with REST APIs for permit lifecycle management.
Implemented validation, workflow-based state management, and database persistence for secure processing.
Streamlined manual approval workflows to improve processing efficiency.


WORKSHOP AND CERTIFICATION
Workshop: 3-Day Workshop on Node.js and Docker by CDW (Sep 2024)
 Paper Presentation: Galaxy’24 at Government Engineering College of Erode (”Transport Permit Website”)

-------------------------
"""

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=resume_context
)

messages = []

while True:
    user_input = input("\nAsk about Hamsa: ")

    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    response = agent.invoke({"messages": messages})

    final_message = response["messages"][-1]
    print("\nAnswer:", final_message.content)

    messages.append({"role": "assistant", "content": final_message.content})
----------------------------------------------------------------------
Langraph:

pip install langgraph langchain langchain-core python-dotenv

