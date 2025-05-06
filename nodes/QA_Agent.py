from langchain_groq import ChatGroq
import os
import json
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
load_dotenv()

class QAAgent:
    def __init__(self):
        self.llm = ChatGroq(api_key=os.environ['GROQ_API_KEY_3'], model='llama-3.3-70b-versatile')
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    def create_QA_agent(self,collection_name,slide_content,lecture_content,question):
        vector_Store= Chroma(persist_directory="chromadb_store",collection_name=collection_name,embedding_function=self.embeddings)
        retriever=vector_Store.as_retriever(search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25})
        
        def retrieve_info(query: str):
            """Retrieves relevant passages from the vector database."""
            docs = retriever.invoke(query)
            print(docs)
            return "\n\n".join([doc.page_content for doc in docs])

        retrieval_tool = Tool(
            name=f"Retrieve",
            func=retrieve_info,
            description=f"Search the vector database to find relevant information."
        )  

        react_agent = initialize_agent(
            tools=[retrieval_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        agent_prompt = f"""
            You are a helpful educational assistant tasked with answering the following question based on lecture and slide materials.

            Lecture Content:
            \"\"\"
            {lecture_content}
            \"\"\"

            Slide Content:
            \"\"\"
            {slide_content}
            \"\"\"

            Question:
            \"\"\"
            {question}
            \"\"\"

            Use the "Retrieve" tool to search the vector database if the answer is not directly obvious from the provided content.

            Think step-by-step. First analyze what the question is asking, then identify if you need to retrieve additional context, and finally construct a clear and accurate answer based on the information.

            Always ensure your answer is grounded in the content and avoids making assumptions.

            Answer:
        """

        agent_response = react_agent.run(agent_prompt)
        return agent_response.strip()

