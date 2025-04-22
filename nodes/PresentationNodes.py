from typing import TypedDict, List, Dict
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain import hub
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import json
from tools.SearchTools import WebSearchTool
from langchain.prompts import ChatPromptTemplate
import asyncio
from utils.scarper import scrape_multiple

load_dotenv()

class PresentationState(TypedDict):
    topic: str
    toc: List[str]
    resources: List[str]
    documents: str
    vector_db: Chroma
    content: Dict[str, str]
    slides: List[Dict[str, str]]
    lecture: List[str]

class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query to find relevant information from the web.")

class Nodes:
    def __init__(self):
        self.llm2 = ChatGroq(api_key=os.environ['GROQ_API_KEY_1'], model='llama-3.3-70b-versatile')
        self.llm = ChatGroq(api_key=os.environ['GROQ_API_KEY_2'], model='llama-3.3-70b-versatile')

        self.web_search_tool = WebSearchTool()
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    def Topic(self, state: PresentationState) -> PresentationState:
        topic = input("Enter the topic you want to study: ")
        state.update({
            "topic": topic,
            "toc": [],
            "resources": [],
            "documents": [],
            "vector_db": {},
            "content": {},
            "slides": [],
            "lecture": [],
        })
        return state

    def SubjectSpecialist(self, state: PresentationState) -> PresentationState:
        topic = state["topic"]
        prompt = f"""
        You are an expert on {topic}. Provide a structured list of five key subtopics for a lecture. The subtopics should be relevant to the main topic and suitable for a comprehensive study plan. Each subtopic should be concise and informative. Make sure the headings of the subtopics you return have enough information to be used as a search query.
        Return them in the following format:
        Subtopic1\nSubtopic2\nSubtopic3\nSubtopic4\nSubtopic5
        """
        response = self.llm.invoke(prompt)
        state["toc"] = response.content.strip().split("\n")[:1]
        return state

    def SearchResources(self, state: PresentationState) -> PresentationState:
        """Searches for resources for all subtopics and collects unique URLs in a single list."""
        url_set = set()

        for subtopic in state["toc"]:
            print(f"Searching for resources on: {subtopic}")

            tools = [self.web_search_tool]
            react_agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )

            agent_prompt = f"""
            You are an intelligent research assistant looking for the best resources to learn about '{subtopic}'.
            Your goal is to find the most relevant and high-quality sources using the web search tool.

            - Start by making an initial search query related to '{subtopic}'.
            - If the results are not satisfactory, refine your query and try again.
            - Stop searching when you have found at most 3 high-quality sources.
            - Only return webpage URLs (no PDFs, courses, or video tutorials).
            - Format: url1\nurl2\nurl3...

            Begin your search now.
            """

            agent_response = react_agent.run(agent_prompt)
            urls = agent_response.strip().split("\n")
            url_set.update(urls)

        state["resources"] = list(url_set)
        print(state["resources"])
        return state

    def ScrapeContent(self,state : PresentationState) -> PresentationState:
        urls=state['resources']
        extracted_texts = asyncio.run(scrape_multiple(urls))
        state["documents"] = extracted_texts
        print("Scraping complete.")
        return state


    def StoreInVectorDB(self, state: PresentationState) -> PresentationState:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splitted_docs = text_splitter.create_documents([state["documents"]])
        state["vector_db"] = Chroma.from_documents(splitted_docs, self.embeddings)
        return state


    def ResearchSpecialist(self, state: PresentationState) -> PresentationState:
        """Uses the single vector database instance for researching all subtopics."""
        if not state["vector_db"]:
            print("Vector database is empty. Skipping research.")
            return state

        retriever = state["vector_db"].as_retriever()

        def retrieve_info(query: str):
            """Retrieves relevant passages from the vector database."""
            docs = retriever.get_relevant_documents(query)
            return "\n\n".join([doc.page_content for doc in docs])

        state["content"] = {}

        for subtopic in state["toc"]:
            print(f"Researching: {subtopic}")

            retrieval_tool = Tool(
                name=f"Retrieve_{subtopic}",
                func=retrieve_info,
                description=f"Search the vector database for {subtopic} to find relevant information."
            )

            react_agent = initialize_agent(
                tools=[retrieval_tool],
                llm=self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )

            agent_prompt = f"""
            You are an expert research assistant generating structured summaries for '{subtopic}'.

            Steps:
            - Use the retrieval tool to fetch the most relevant information from the vector database.
            - Extract the most useful insights.
            - Structure the information into a well-organized summary.
            - Remove redundant, irrelevant, or poorly formatted parts.
            - Ensure clarity and readability.

            Begin now.
            """

            agent_response = react_agent.run(agent_prompt)
            state["content"][subtopic] = agent_response.strip()

        print("Researching Complete")
        return state



    def SlidesMaker(self, state: PresentationState) -> PresentationState:
        content = state['content']
        slides = []
        try:
            for key,value in content.items():
                template = '''
                    Create slides for the subtopic: {topic}. Ensure they feel like continuation of a lecture.
                    **Instructions:**
                    1. Include coding examples with explanations if required. Retain all important details.
                    2. Use JSON format: `title`, `content`, and `code` (use `""` if not applicable).
                    3. Include only relevant information. For example, if the topic is an introduction, do not add conclusions.
                    4. Return a list of JSON objects without additional text.
                    5- Most important the content of the slides should be long atleast 8 lines.
                    6- Don't repeat the same title for different slides. If the title is already present in the list below.{slides_title}

                    **Topic and Information:**
                    Topic: {topic}
                    Information: {information}

                    **Example Output:**
                    [
                        {{
                            "title": "Contains the title of the slide",
                            "content": "Contains the text content of the slide",
                            "code": "Contains any code if programming related thing, mathematical work if related to math, etc."
                        }},
                    ]
                '''
                prompt = ChatPromptTemplate.from_template(template)
                slides_title=[slide['title'] for slide in slides]
                message = prompt.invoke({'topic': key, 'information': value,'slides_title':slides_title})
                slides_content = self.llm2.invoke(message).content
                slides_json = json.loads(slides_content)
                slides.append(slides_json)
        except Exception as e:
            print(e)
        finally:
            print("Slides Making Complete")
            state["slides"] = slides
            return state


    def LectureAgent(self, state: PresentationState) -> PresentationState:
        slides=state['slides']
        topic=state['topic']
        for item in slides:
            template = '''
                Create a teaching script for the topic: {topic} based on the provided slides.
                **Instructions:**
                1. Explain and expand on slide content; do not read slides verbatim.
                2. For coding examples, explain what the code does, its purpose, and how it works.
                3. Use real-world examples, scenarios, or analogies to explain concepts.
                4. Provide clear, detailed explanations to ensure students understand the "why" and "how."
                5. Make the lecture engaging with rhetorical questions or critical thinking prompts.

                **Topic and Slides:**
                Topic: {topic}
                Slides: {slides}

                **Output:**
                - A detailed teaching script divided into sections corresponding to the slides.
                - Explanations, examples, and real-world applications for each slide.
            '''
            prompt=ChatPromptTemplate.from_template(template)

            message=prompt.invoke({'topic':topic,'slides':slides})

            lecture_content = self.llm2.invoke(message)
            lecture_content = lecture_content.content.strip()
            state['lecture'].append(lecture_content)
        print("Lecture Agent Complete")
        return state


