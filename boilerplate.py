import operator
import re
import os
import time
import logging
from datetime import datetime
from typing import Annotated, TypedDict, List
from Bio import Entrez
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# --- Logging Setup ---
_log_file = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Silence noisy third-party HTTP loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger.info("Logging initialized. Log file: %s", _log_file)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    pubmed_results: List[str]
    current_search: str

# --- 1. NEW NODE: Query Refiner ---
llm = ChatOllama(model="deepseek-r1:7b", temperature=0)

def query_refiner_node(state: AgentState):
    logger.info("[NODE: QUERY_REFINER] Optimizing search terms...")
    user_input = state["messages"][0].content
    logger.debug("Original user input: %s", user_input)

    # We ask the LLM to create a professional PubMed Boolean query
    refine_prompt = f"""Convert the following user request into a concise PubMed search query using Boolean operators (AND, OR) and medical keywords.
    User Request: {user_input}
    Output ONLY the search string."""

    refined_query = llm.invoke(refine_prompt).content.strip()
    # Clean up any deepseek thinking tags if they leak into the query
    if "</think>" in refined_query:
        refined_query = refined_query.split("</think>")[-1].strip()

    logger.info("Query refinement complete. Original: '%s...' -> Optimized: '%s'", user_input, refined_query)
    return {"current_search": refined_query}

# --- 2. TOOL: PubMed ---
def pubmed_tool(state: AgentState):
    logger.info("[NODE: PUBMED_TOOL] Searching PubMed database...")
    Entrez.email = "csiniscalchi@uchicago.edu"
    query = state["current_search"]
    logger.debug("PubMed query: %s", query)

    h_search = Entrez.esearch(db="pubmed", term=query, retmax=5)
    ids = Entrez.read(h_search)["IdList"]
    logger.debug("PubMed returned IDs: %s", ids)

    if not ids:
        logger.warning("[NODE: PUBMED_TOOL] No results found for query: '%s'", query)
        return {"pubmed_results": []}

    h_fetch = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
    abstracts = h_fetch.read()
    logger.info("[NODE: PUBMED_TOOL] Retrieved %d abstracts.", len(ids))
    return {"pubmed_results": [abstracts]}

# --- 3. BRAIN: Researcher Node (Strict Grounding) ---
def researcher_node(state: AgentState):
    logger.info("[NODE: RESEARCHER] Synthesizing findings from context...")
    results = state.get("pubmed_results", [])
    context = "\n".join(results)

    if not context:
        logger.warning("[NODE: RESEARCHER] No PubMed context available. Proceeding with empty context.")
    else:
        logger.debug("[NODE: RESEARCHER] Context length: %d characters.", len(context))

    # If context is empty, we force the agent to admit it or ask for a new search
    system_msg = f"""You are a Clinical Oncology Assistant.
    DATA CONTEXT:
    {context if context else "NO DATA FOUND IN PUBMED."}

    STRICT RULES:
    1. If CONTEXT is "NO DATA FOUND", you MUST say "I cannot find this information in PubMed. Please refine your search."
    2. DO NOT use your internal training memory to name mutations.
    3. Every claim MUST cite a PMID from the context.
    4. Format your answer clearly."""

    response = llm.invoke([HumanMessage(content=system_msg)] + state["messages"])
    logger.info("[NODE: RESEARCHER] Response generated. Length: %d characters.", len(response.content))
    return {"messages": [response]}

# --- 4. HARD CHECK: Reference Check ---
def reference_check_node(state: AgentState):
    logger.info("[NODE: REFERENCE_CHECK] Auditing citations...")
    last_msg = state["messages"][-1].content
    context = "\n".join(state.get("pubmed_results", []))

    cited = re.findall(r"PMID:?\s*(\d+)", last_msg)
    available = re.findall(r"PMID:?\s*(\d+)", context)
    logger.debug("Cited PMIDs: %s | Available PMIDs: %s", cited, available)

    hallucinated = [p for p in cited if p not in available]

    if hallucinated:
        logger.error("[NODE: REFERENCE_CHECK] FAILED - Fabricated PMIDs detected: %s", hallucinated)
        return {"messages": [HumanMessage(content=f"REJECT: The PMIDs {hallucinated} are not in the provided source data.")]}

    logger.info("[NODE: REFERENCE_CHECK] PASSED - All %d cited PMIDs verified.", len(cited))
    return {"messages": [HumanMessage(content="REF_CHECK_PASSED")]}

# --- 5. JUDGE: Validator Node ---
def validator_node(state: AgentState):
    logger.info("[NODE: VALIDATOR] Running final logic check...")
    last_answer = state["messages"][-2].content
    context = "\n".join(state.get("pubmed_results", []))
    logger.debug("[NODE: VALIDATOR] Validating answer of length %d against context.", len(last_answer))

    val_prompt = f"Does the following answer contain any info NOT in the context?\nContext: {context}\nAnswer: {last_answer}\nRespond VALID or REJECT: [reason]."

    response = llm.invoke(val_prompt)
    verdict = response.content.strip()
    if verdict.upper().startswith("REJECT"):
        logger.warning("[NODE: VALIDATOR] REJECTED - %s", verdict)
    else:
        logger.info("[NODE: VALIDATOR] VALID - %s", verdict)
    return {"messages": [response]}

# --- 6. GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

workflow.add_node("query_refiner", query_refiner_node)
workflow.add_node("pubmed_tool", pubmed_tool)
workflow.add_node("researcher", researcher_node)
workflow.add_node("ref_check", reference_check_node)
workflow.add_node("validator", validator_node)

# Flow: START -> Refine -> Search -> Research -> Check -> Validate
workflow.add_edge(START, "query_refiner")
workflow.add_edge("query_refiner", "pubmed_tool")
workflow.add_edge("pubmed_tool", "researcher")

def post_research_router(state):
    # If the researcher outputted a new SEARCH: command, go back to tool
    if "SEARCH:" in state["messages"][-1].content:
        logger.info("[ROUTER: post_research] Routing back to pubmed_tool (new search requested).")
        return "pubmed_tool"
    logger.info("[ROUTER: post_research] Routing to ref_check.")
    return "ref_check"

def post_check_router(state):
    msg = state["messages"][-1].content
    if "REJECT:" in msg:
        logger.info("[ROUTER: post_check] Reference check failed. Routing back to researcher.")
        return "researcher"
    logger.info("[ROUTER: post_check] Reference check passed. Routing to validator.")
    return "validator"

def post_val_router(state):
    if "REJECT:" in state["messages"][-1].content.upper():
        logger.info("[ROUTER: post_val] Validation failed. Routing back to researcher.")
        return "researcher"
    logger.info("[ROUTER: post_val] Validation passed. Routing to END.")
    return END

workflow.add_conditional_edges("researcher", post_research_router)
workflow.add_conditional_edges("ref_check", post_check_router)
workflow.add_conditional_edges("validator", post_val_router)

app = workflow.compile()

if __name__ == "__main__":
    query = input("\n[User]: ")
    logger.info("Starting verified workflow. User query: '%s'", query)
    app.invoke({"messages": [HumanMessage(content=query)], "pubmed_results": []})
    logger.info("Workflow complete.")