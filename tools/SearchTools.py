from pydantic import BaseModel, Field
from typing import Type
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
import requests
import os

load_dotenv()


class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query to find relevant information from the web.")

class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "This tool uses the Google Serper API to fetch relevant information from the web based on the provided query."
    args_schema: Type[BaseModel] = WebSearchArgs

    def _run(self, query: str):
        api_key = os.environ['SERPER_API_KEY']
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key}
        params = {"q": query}

        response = requests.get(url, headers=headers, params=params)
        search_results = response.json()
        results = [{item["link"]:item['snippet']} for item in search_results.get("organic", [])[:5]]
        return results

