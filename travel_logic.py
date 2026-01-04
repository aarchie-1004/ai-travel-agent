# travel_logic.py (Adapted from career_logic.py)
import os
import datetime
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
import requests
import json
import logging
from rich.logging import RichHandler
from rich.console import Console

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# --- Configure Rich Logging ---
LOG_LEVEL = "INFO"
logging.basicConfig(
    level=LOG_LEVEL, format="%(message)s", datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)]
)
logger = logging.getLogger("rich")
console = Console()
# --- End Logging Config ---

load_dotenv()

# --- Initialize LLMs and Get API Keys ---
groq_api_key = os.getenv("GROQ_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

if not groq_api_key or not serper_api_key:
    logger.error("[bold red]API keys (GROQ_API_KEY, SERPER_API_KEY) not found in .env file.[/]")
    raise ValueError("API keys (GROQ_API_KEY, SERPER_API_KEY) not found in .env file.")

try:
    # LLM for extracting key travel criteria
    extraction_llm = ChatGroq(
        groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.0
    )
    # LLM for generating travel ideas (can be the same or different)
    planning_llm = ChatGroq(
        groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.7
    )
    logger.info("[bold green]:robot: Groq LLMs initialized successfully.[/]")
except Exception as e:
    logger.error(f"[bold red]Could not initialize Groq LLMs: {e}[/]", exc_info=True)
    raise

# --- Define Graph State ---
class GraphState(TypedDict):
    """
    Represents the state of our travel graph.
    """
    original_input: str
    cleaned_criteria: Optional[str] = None # Renamed from cleaned_topic
    search_results: Optional[str] = None
    travel_suggestions: Optional[str] = None # Renamed from final_roadmap
    error: Optional[str] = None

# --- Define Node Functions ---

def clean_criteria_node(state: GraphState) -> GraphState:
    """
    Cleans the user input to extract key travel criteria (destination, interests, budget, duration).
    """
    logger.info("[bold blue]>> Entering Node: clean_criteria[/]")
    user_input = state['original_input']
    logger.info(f"   Input: '[italic]{user_input}[/italic]'")
    try:
        # Updated system prompt for travel
        system_prompt = SystemMessage(content="You are an expert travel query parser. Extract key details like destination/area, interests (e.g., beach, mountains, history, food), budget (e.g., budget, luxury), and duration (e.g., weekend, week) as a concise comma-separated list. If no specific location is mentioned, note that. Example Input: looking for a budget weekend trip near Ludhiana, interested in history. Example Output: budget, weekend trip, near Ludhiana, history")
        user_prompt = HumanMessage(content=user_input)
        logger.info("   Calling Extraction LLM...")
        response = extraction_llm.invoke([system_prompt, user_prompt])
        cleaned_criteria = response.content.strip()
        if not cleaned_criteria:
             logger.warning("  Cleaned criteria was empty, falling back to original input.")
             cleaned_criteria = user_input # Fallback
        logger.info(f"   [green]Cleaned criteria:[/green] '[italic]{cleaned_criteria}[/italic]'")
        return {"cleaned_criteria": cleaned_criteria, "error": None}
    except Exception as e:
        logger.error(f"  [bold red]Error cleaning criteria:[/bold red] {e}", exc_info=True)
        return {"error": f"Failed to clean criteria: {e}", "cleaned_criteria": user_input} # Fallback

def search_travel_ideas_node(state: GraphState) -> GraphState:
    """
    Searches the web for travel ideas based on the cleaned criteria.
    """
    logger.info("[bold blue]>> Entering Node: search_travel_ideas[/]")
    if state.get("error"):
        logger.warning("[yellow]  Skipping search due to previous error.[/]")
        return {}
    cleaned_criteria = state['cleaned_criteria']
    if not cleaned_criteria:
        logger.error("[bold red]  Cannot search without cleaned criteria.[/]")
        return {"error": "Cannot search without cleaned criteria."}

    # Construct a search query focused on travel ideas
    search_query = f"travel destinations or ideas for '{cleaned_criteria}'"
    # Optional: Add current year if relevant f"travel ideas for '{cleaned_criteria}' in 2025"
    logger.info(f"   Search query: '[italic]{search_query}[/italic]'")

    headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
    payload = json.dumps({"q": search_query})
    api_url = "https://google.serper.dev/search"

    try:
        logger.info(f"   [bold yellow]:satellite: Calling Serper API:[/bold yellow] {api_url}")
        response = requests.post(api_url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        results = response.json()

        # Format results (focus on destinations, attractions, summaries)
        formatted_results = f"Search Query: {search_query}\n\n"
        if results.get("answerBox"):
             formatted_results += f"[bold]Direct Answer/Summary:[/bold]\n"
             ab = results["answerBox"]
             if ab.get('title'): formatted_results += f"  Title: {ab['title']}\n"
             if ab.get('answer'): formatted_results += f"  Answer: {ab['answer']}\n"
             elif ab.get('snippet'): formatted_results += f"  Snippet: {ab['snippet']}\n"
             if ab.get('link'): formatted_results += f"  Link: {ab['link']}\n"
             formatted_results += "---\n"
        if results.get("knowledgeGraph"): # Knowledge graph often has good place summaries
            formatted_results += "[bold]Place Summary (Knowledge Graph):[/bold]\n"
            kg = results["knowledgeGraph"]
            if kg.get('title'): formatted_results += f"  Title: {kg['title']}\n"
            if kg.get('description'): formatted_results += f"  Description: {kg['description']}\n"
            formatted_results += "---\n"
        if results.get("organic"):
            formatted_results += "[bold]Top Web Results:[/bold]\n"
            for i, item in enumerate(results["organic"][:4], 1): # Get top 4 results
                formatted_results += f"{i}. {item.get('title', 'N/A')}\n"
                if item.get('snippet'): formatted_results += f"   Snippet: {item['snippet'][:250]}...\n" # Slightly longer snippets
        if len(formatted_results) < len(f"Search Query: {search_query}\n\n") + 20:
             formatted_results += "[yellow]No particularly relevant travel search results found.[/yellow]"

        logger.info("   [green]Serper search successful.[/green]")
        logger.debug(f" Full formatted search results sent to LLM:\n{formatted_results}")
        return {"search_results": formatted_results, "error": None}

    # ... (Error handling remains the same) ...
    except requests.exceptions.Timeout:
        logger.error("  [bold red]Error: Serper API request timed out.[/]", exc_info=True)
        return {"error": "Search request timed out.", "search_results": "Search timed out."}
    except requests.exceptions.RequestException as e:
        logger.error(f"  [bold red]Error during direct Serper API call:[/bold red] {e}", exc_info=True)
        error_detail = ""
        try: error_detail = response.json().get('message', '') if response else ''
        except: pass
        return {"error": f"Serper API request failed: {e}. {error_detail}", "search_results": "Search failed."}
    except Exception as e:
         logger.error(f"  [bold red]Error processing Serper response:[/bold red] {e}", exc_info=True)
         return {"error": f"Failed processing Serper results: {e}", "search_results": "Search failed."}


def generate_travel_ideas_node(state: GraphState) -> GraphState:
    """
    Generates travel suggestions using the LLM based on criteria and search results.
    """
    logger.info("[bold blue]>> Entering Node: generate_travel_ideas[/]")
    if state.get("error"):
        logger.warning("[yellow]  Skipping suggestions due to previous error.[/]")
        return {}
    user_input = state['original_input']
    cleaned_criteria = state['cleaned_criteria']
    search_results = state['search_results']

    if not cleaned_criteria or not search_results:
        logger.error("[bold red]  Missing cleaned criteria or search results for suggestion generation.[/]")
        return {"error": "Missing info for suggestion generation."}

    # --- MODIFIED SYSTEM PROMPT for Travel ---
    system_prompt_content = f"""
    You are a friendly and enthusiastic AI travel advisor! Like a helpful friend sharing exciting ideas.
    Your goal is to suggest 2-3 interesting travel destinations or trip ideas based on the user's request and relevant web search results.
    Use a warm, encouraging, and slightly conversational tone (e.g., "Okay, let's see!", "How about this?", "Sounds like fun!").

    Instructions:
    1.  Start with a friendly greeting acknowledging the user's request (e.g., "Hey! Looking for ideas based on '{cleaned_criteria}'? Let's explore!").
    2.  Analyze the user's original input ('{user_input}') and the cleaned travel criteria ('{cleaned_criteria}').
    3.  Carefully consider the provided web search results for inspiration and details.
    4.  Suggest 2-3 specific destinations or trip concepts.
    5.  For EACH suggestion, provide details in Markdown:
        * Start with an engaging sentence (e.g., "How about exploring [Place]?").
        * **Why it fits:** Briefly explain why this idea matches the user's criteria (e.g., "It's known for its amazing beaches, perfect for relaxing!").
        * **Things to Do/See:** Mention 2-4 key attractions, activities, or highlights based on the search results or general knowledge.
        * **Vibe:** Briefly describe the general feel (e.g., "Relaxing beach escape," "Adventurous mountain trek," "Historic city exploration").
        * (Optional) Mention budget/duration fit if easily inferred from criteria/search.
    6.  Conclude the entire response with a friendly closing remark (e.g., "Hope these give you some inspiration! Let me know if you want to dive deeper into one!").
    7.  **IMPORTANT:** Format clearly using Markdown headings (## or ###) for each suggestion. Use bullet points. DO NOT include the raw web search results in your final response.
    """
    # --- END MODIFIED SYSTEM PROMPT ---

    user_prompt_content = f"User's Original Input: {user_input}\nCleaned Criteria: {cleaned_criteria}\n\nFormatted Search Results (Use these for context, DO NOT REPEAT them):\n---\n{search_results}\n---\nOkay, give me some exciting travel ideas based on this!"

    system_prompt = SystemMessage(content=system_prompt_content)
    user_prompt_final = HumanMessage(content=user_prompt_content)

    logger.info("  Calling Planning LLM for travel ideas...")
    try:
        response = planning_llm.invoke([system_prompt, user_prompt_final])
        final_suggestions = response.content.strip()
        if not final_suggestions:
             logger.warning("  LLM returned empty travel suggestions.")
             return {"error": "The AI planner didn't generate any travel ideas. Please try rephrasing your request."}

        logger.info("  [green]Travel suggestions generated successfully.[/green]")
        logger.debug(f"  Generated Suggestions:\n{final_suggestions}")
        # Rename the output key in the state
        return {"travel_suggestions": final_suggestions, "error": None}
    except Exception as e:
        logger.error(f"  [bold red]Error generating travel suggestions:[/bold red] {e}", exc_info=True)
        return {"error": f"Failed to generate final travel suggestions: {e}"}


# --- Define and Compile Graph --- (Update node names and final edge)
workflow = StateGraph(GraphState)
workflow.add_node("clean_criteria", clean_criteria_node)
workflow.add_node("search_travel_ideas", search_travel_ideas_node)
workflow.add_node("generate_travel_ideas", generate_travel_ideas_node)

workflow.set_entry_point("clean_criteria")
workflow.add_edge("clean_criteria", "search_travel_ideas")
workflow.add_edge("search_travel_ideas", "generate_travel_ideas")
workflow.add_edge("generate_travel_ideas", END) # End after suggestions

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)
logger.info("[bold green]:airplane: Travel LangGraph compiled successfully.[/]")

# --- Function to Get Graph ---
def get_runnable_graph():
    return app_graph