from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from typing import Type, ClassVar
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
import requests
import wikipedia
import os

load_dotenv()

class DocumentQueryArgs(BaseModel):
    query: str = Field(description="The query to search in the vector database.")

class DocumentQueryTool(BaseTool):
    name: str = "Document Query Tool"
    description: str = "This tool queries a ChromaDB vector database using LangChain to find relevant documents based on the provided query about C++ programming fundamentals."
    args_schema: Type[BaseModel] = DocumentQueryArgs
    embedder: ClassVar[HuggingFaceEmbeddings] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store: ClassVar[Chroma] = Chroma(
        persist_directory="chromadb_store",
        collection_name="document_store",
        embedding_function=embedder
    )

    def _run(self, query: str):
        results = self.vector_store.similarity_search(query, k=6)
        return [result.page_content for result in results]

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
        print(search_results)
        results = [item["snippet"] for item in search_results.get("organic", [])[:5]]
        return results

class WikipediaSearchArgs(BaseModel):
    query: str = Field(description="The search query to find relevant information from Wikipedia.")

class WikipediaSearchTool(BaseTool):
    name: str = "Wikipedia Search Tool"
    description: str = "This tool searches Wikipedia for relevant information based on the provided query."
    args_schema: Type[BaseModel] = WikipediaSearchArgs

    def _run(self, query: str):
        try:
            summary = wikipedia.summary(query, sentences=3)
            return summary
        except wikipedia.exceptions.PageError:
            return "No Wikipedia page found for the query."
