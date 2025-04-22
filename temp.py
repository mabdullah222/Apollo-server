from typing import TypedDict, List, Dict
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
import os
import json
from bs4 import BeautifulSoup
from tools.SearchTools import WebSearchTool

load_dotenv()

class PresentationState(TypedDict):
    topic: str
    toc: List[str]
    resources: Dict[str, List[str]]
    documents: Dict[str, List[str]]
    vector_db: Dict[str, FAISS]
    content: List[Dict[str, str]]
    slides: List[Dict[str,str]]
    lecture: List[str]  

class Nodes:
    def __init__(self):
        self.llm = ChatGroq(api_key=os.environ['GROQ_API_KEY_3'],model='llama-3.3-70b-versatile')
        self.web_search_tool = WebSearchTool()

        self.react_tools = [
            self.web_search_tool
        ]

        # Initialize ReAct agent
        self.react_agent = initialize_agent(
            tools=self.react_tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def Topic(self, state: PresentationState) -> PresentationState:
        topic = input("Enter the topic you want to study: ")
        state.update({
            "topic": topic,
            "toc": [],
            "resources": {},
            "documents": {},
            "vector_db": {},
            "content": [],
            "slides": [],
            "lecture": [],
        })
        return state

    def SubjectSpecialist(self, state: PresentationState) -> PresentationState:
        topic = state["topic"]
        prompt = f"""
        You are an expert on {topic}. Provide a structured list of five key subtopics for a lecture.
        Return them in the following format:
        Subtopic1\nSubtopic2\nSubtopic3\nSubtopic4\nSubtopic5
        """
        response = self.llm.invoke(prompt)
        state["toc"] = response.content.strip().split("\n")
        return state
    
    def SearchResources(self, state: PresentationState) -> PresentationState:
        """Searches for URLs related to each subtopic."""
        state["resources"] = {}
        for subtopic in state["toc"]:
            print(f"Searching for resources on: {subtopic}")
            query = f"Best resources for {subtopic}"
            results = self.web_search_tool._run(query)
            state["resources"][subtopic] = results
        return state
    
    def ScrapeContent(self, state: PresentationState) -> PresentationState:
        """Scrapes content from gathered URLs."""
        state["documents"] = {}
        for subtopic, urls in state["resources"].items():
            extracted_texts = []
            for url in urls:
                try:
                    response = requests.get(url)
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = ' '.join([p.text for p in soup.find_all("p")])
                    extracted_texts.append(text)
                except Exception as e:
                    print(f"Error scraping {url}: {e}")
            state["documents"][subtopic] = extracted_texts
        return state
    
    def StoreInVectorDB(self, state: PresentationState) -> PresentationState:
        """Stores scraped content into a vector database."""
        state["vector_db"] = {}
        for subtopic, docs in state["documents"].items():
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splitted_docs = text_splitter.create_documents(docs)
            state["vector_db"][subtopic] = FAISS.from_documents(splitted_docs, self.embeddings)
        return state
    
    def ResearchSpecialist(self, state: PresentationState) -> PresentationState:
        """Retrieves knowledge from vector DB instead of free LLM responses."""
        for subtopic in state["toc"]:
            print(f"Researching: {subtopic}")
            retriever = state["vector_db"][subtopic].as_retriever()
            qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=retriever)
            research_result = qa_chain.run(f"Provide detailed research on {subtopic}.")
            state["lecture"].append(research_result)
        return state

    def ResearchSpecialist(self, state: PresentationState) -> PresentationState:
        for current_task in state["toc"]:
            print("The current task is to search: ",current_task)
            research_prompt = f'''Conduct in-depth research on the topic: "{current_task}".
        **Instructions to Follow Strictly:**
        1. **Relevance**: Focus only on information directly related to the topic. Exclude tangential or unrelated details.
        2. **Comprehensiveness**: Provide a complete overview of the topic, including key concepts, definitions, and applications.
        3. **Coding Examples (if applicable)**: If the topic involves programming or technical implementation, include a functional, well-documented code example that demonstrates the concept.
        4. **Accuracy**: Ensure all information is accurate, up-to-date, and sourced from reliable references.
        5. **Clarity**: Present the information in a clear, concise, and structured manner for easy understanding.
        6. **Actionability**: Ensure the research results are actionable and directly usable for the presentation or further processing.
        **Deliverables:**
        - A detailed summary of the topic.
        - Key points, facts, and insights.
        - A code example (if applicable).
        - References or sources used for the research.
        **Note**: Do not include information about other topics in the TOC. Each topic will be handled separately.
        '''

            research_result = self.react_agent.run(research_prompt)

            state["information"].append({current_task: research_result})

        return state

    def ContentWriter(self, state: PresentationState) -> PresentationState:
        topic=state['topic']
        content={}
        information=state['information']
        try:
            for entry in information:
                for key,value in entry.items():
                    template = '''
                        Create detailed, structured notes for students on: {topic}.
                        **Instructions:**
                        1. Expand and clarify the collected information. Do not omit or reduce details.
                        2. Stay strictly on topic. If the topic is an introduction, only provide an introduction.
                        3. Use headings, bullet points, and clear explanations for a student-friendly format.
                        4. Include background information if necessary for better understanding.
                        5. Ensure all content is directly relevant to the topic.

                        **Information Collected:**
                        {information}

                        **Output:**
                        - Well-structured notes with clear explanations and examples.
                    '''
                    prompt=ChatPromptTemplate.from_template(template)

                    message=prompt.invoke({'topic':key,'information':value})
                    processed_content = self.llm.invoke(message).content.strip()
                    content[key]=processed_content
        except Exception as e:
            print(e)
            with open('output.json','w') as ifile:
                json.dump(state,ifile)
        finally:
            state["content"] = content
            return state
        
    def SlidesMaker(self, state: PresentationState) -> PresentationState:
        content = state['content']
        slides = []
        try:
            for key,value in content.items():
                template = '''
                    Create slides for a lecture on: {topic}. Ensure they feel like part of a larger lecture.
                    **Instructions:**
                    1. Maintain continuity with the larger lecture.
                    2. Include coding examples with explanations. Retain all important details.
                    3. Divide content into smaller parts, each corresponding to one slide.
                    4. Use JSON format: `title`, `content`, and `code` (use `""` if not applicable).
                    5. Include only relevant information. For example, if the topic is an introduction, do not add conclusions.
                    6. Return a list of JSON objects without additional text.

                    **Topic and Information:**
                    Topic: {topic}
                    Information: {information}

                    **Example Output:**
                    [
                        {{
                            "title": "Introduction to Pointers",
                            "content": "Pointers are a core concept in C++ that allow for direct memory manipulation...",
                            "code": ""
                        }},
                        {{
                            "title": "",
                            "content": "In C++, declare a pointer with the asterisk (*) symbol. Example:",
                            "code": "int *p;  // Declares a pointer to an integer\nint a = 10;\nint *p = &a;"
                        }}
                    ]
                '''
                prompt = ChatPromptTemplate.from_template(template)
                message = prompt.invoke({'topic': key, 'information': value})
                slides_content = self.llm.invoke(message).content
                slides_json = json.loads(slides_content)
                slides.append(slides_json)
        except Exception as e:
            print(e)
            with open('output.json', 'w') as ifile:
                json.dump(state, ifile)
        finally:
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

            lecture_content = self.llm.invoke(message)
            lecture_content = lecture_content.content.strip()
            state['lecture'].append(lecture_content)
        return state


