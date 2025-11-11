import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from gemini_client import GeminiClient
# from IPython.display import Image, display
# from mcp_client import MCPClient  # your GitHub MCP wrapper


class InterviewAgentState(TypedDict):
    """State schema for the interview agent workflow."""
    job_description: str
    resume_text: str
    github_user: str
    worked_topics: List[str]
    resume_skills: List[str]  # Technical and non-technical skills from resume
    required_topics: List[str]
    missing_topics: List[str]
    overlap_topics: List[str]
    learning_plan: str
    tuned_resume: str

gemini = GeminiClient()
# mcp = MCPClient(base_url=os.getenv("MCP_URL"), auth_token=os.getenv("MCP_TOKEN"))

async def github_analysis_node(state: InterviewAgentState):
    user = state["github_user"]
    # Example: list repos, pick language/tech metadata
    # repos = await mcp.list_user_repos(user)
    # ... fetch each repo, search code, analyze topics (you can embed chunk + cluster later)
    worked_topics = ["Kafka Streams", "NestJS + Mongoose", "PDF Text/Image extraction"]  # placeholder
    return {"worked_topics":worked_topics}

async def resume_expert_node(state: InterviewAgentState):
    """
    Resume expert from top IT company reviews resume and extracts:
    - Technical skills (languages, frameworks, tools, databases)
    - Non-technical skills (leadership, communication, project management)
    """
    resume = state["resume_text"]
    prompt = f"""You are a senior hiring manager and resume expert from a top IT company.

Extract concise skills (technical and non-technical) from the resume.
Output requirements:
- One skill per line
- Each skill: short phrase (max 5 words)
- No headings, no explanations, no numbering
- Max 40 skills

Resume:
{resume}

Skills:"""
    
    skills_text = await gemini.generate(prompt)
    # Parse skills into list (clean up whitespace)
    resume_skills = [skill.strip() for skill in skills_text.split("\n") if skill.strip()]
    return {"resume_skills": resume_skills}

async def jd_extraction_node(state):
    jd = state["job_description"]
    prompt = f"""Extract the key required skills and topics from the following job description.
Return a concise list: one short phrase per line (max 5 words each). No explanations or headings.

{jd}"""
    topics_text = await gemini.generate(prompt)
    # parse topics_text into list
    required_topics = topics_text.split("\n")  # crude
    return {"required_topics": required_topics}

def gap_analysis_node(state):
    # Combine skills from both GitHub analysis and resume expert review
    github_topics = set(state["worked_topics"])
    resume_skills = set(state["resume_skills"])
    worked = github_topics.union(resume_skills)  # All skills from both sources
    
    required = set(state["required_topics"])
    missing = list(required - worked)
    overlap = list(required & worked)
    return {"missing_topics": missing, "overlap_topics": overlap}

async def learning_plan_node(state):
    missing = state["missing_topics"]
    prompt = f"""You are a learning coach with IT industry experience.
Produce a concise learning plan to cover these topics.
Output format:
- Numbered steps (1., 2., ...)
- Each step should be a short action or resource (max 10 words)
- Limit to 6 steps
No additional explanation.

Topics: {missing}
"""
    plan = await gemini.generate(prompt)
    return {"learning_plan": plan}

async def resume_tuning_node(state):
    resume = state["resume_text"]
    overlap = state["overlap_topics"]
    prompt = f"""You are a career coach with IT industry experience.
Rewrite the resume concisely to highlight the candidate's strengths: {overlap}.
Requirements:
- Keep bullets short (<= 15 words)
- Emphasize measurable achievements and relevant keywords from the job description
- Return only the revised resume text (no commentary)

Job description context:
{state['job_description']}

Resume:
{resume}
"""
    tuned = await gemini.generate(prompt)
    return {"tuned_resume": tuned}

def build_graph():
    """Build the interview agent workflow graph using LangGraph 1+."""
    print("Building graph...")
    graph = StateGraph(InterviewAgentState)
    
    # Add nodes
    graph.add_node("github_analysis", github_analysis_node)
    graph.add_node("resume_expert", resume_expert_node)
    graph.add_node("jd_extraction", jd_extraction_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("learning_plan", learning_plan_node)
    graph.add_node("resume_tuning", resume_tuning_node)
    
    # Add edges - parallel analysis from START
    graph.add_edge(START, "github_analysis")
    graph.add_edge(START, "resume_expert")
    graph.add_edge(START, "jd_extraction")
    
    # Both analysis nodes feed into gap_analysis
    graph.add_edge("github_analysis", "gap_analysis")
    graph.add_edge("resume_expert", "gap_analysis")
    graph.add_edge("jd_extraction", "gap_analysis")
    
    # Gap analysis outputs to learning and tuning
    graph.add_edge("gap_analysis", "learning_plan")
    graph.add_edge("gap_analysis", "resume_tuning")
    
    # Set end edges
    graph.add_edge("learning_plan", END)
    graph.add_edge("resume_tuning", END)
    
    # Compile and return
    graph = graph.compile()
    print("Graph compiled successfully.")
    # display(Image(graph.get_graph().draw_mermaid_png()))
    return graph
