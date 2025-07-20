print("Script started...")

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Set Gemini API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Agent to search the web
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Gemini(id="gemini-1.5-flash"),  # or "gemini-1.5-pro"
    tools=[DuckDuckGoTools()],
    instructions=["Always include source"],
    show_tool_calls=True,
    markdown=True
)

# Agent to get financial data
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Gemini(id="gemini-1.5-flash"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True
    )],
    instructions=["Display data in table format"],
    show_tool_calls=True,
    markdown=True
)

# Multi-agent system
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Gemini(id="gemini-1.5-flash"),
    instructions=["Always include source", "Display data in table format"],
    show_tool_calls=True,
    markdown=True
)


