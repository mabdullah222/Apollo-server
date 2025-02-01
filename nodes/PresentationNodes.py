from typing import TypedDict, List, Dict
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from tools.SearchTools import DocumentQueryTool,WebSearchTool,WikipediaSearchTool
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from dotenv import load_dotenv
import os
import time
import json
from langgraph.graph import END


load_dotenv()

class PresentationState(TypedDict):
    topic: str
    toc: List[str]
    information: List[Dict[str, str]]
    content: List[Dict[str, str]]
    slides: List[Dict[str,str]]
    lecture: List[str]

class Nodes:
    def __init__(self):
        self.llm = ChatGroq(api_key=os.environ['GROQ_API_KEY_3'],model='llama-3.3-70b-versatile')
        # self.llm=ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'],model='gpt-4o')
        self.document_tool = DocumentQueryTool()
        self.web_search_tool = WebSearchTool()
        self.wikipedia_tool = WikipediaSearchTool()

        self.react_tools = [
            self.document_tool,
            self.web_search_tool,
            # self.wikipedia_tool
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
            "information": [],
            "content": {},
            "slides": [],
            "lecture":[],
        })
        return state

    def SubjectSpecialist(self, state: PresentationState) -> PresentationState:
        topic = state["topic"]
        prompt = (
            f"You are a subject matter expert. Create a structured plan to teach the topic '{topic}'. "
            "Divide this topic into five sub-topics.You have to choose only the five most important sub-topics.List the essential elements needed to understand it, such as an introduction, examples, use cases,coding examples of the list of elements, and applications.The result should only be the table of contents seperated by commas.No additional information is required."
        )
        response = self.llm.invoke(prompt)
        state["toc"] = [f"{item.strip()}" for item in response.content.split(",") if item.strip()]
        return state

    def ResearchSpecialist(self, state: PresentationState) -> PresentationState:
        for current_task in state["toc"]:
            print("The current task is to search: ",current_task)
            research_prompt = f'''Research the topic: {current_task}.
            Instructions to strictly follow:
            1- Use the available tools to gather information.
            2- All the information about the research topic should be returned.
            3- If the research topic required to return a coding example then the final example should contain code example.
            4- Make sure that the final answer fulifills the requirement of the research topic.
            5- Only return the information which is relevant to the research topic. Beacause other topics would be handled by their respective search. 
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
                        You are a teacher assitant and you have collected information collected information for you to teach the topic: {topic}.
                        Instructions:
                        1- Add the details to the information.
                        2- Dont reduce the information instead add upon it to make it more clear,coherent and correct.
                        3- Strictly stay on the topic if the topic is about introduction of something give only introduction dont add other stuff like conclusion or something.
                        3- Bring it in the format so it can serve as notes for the students who want to study the topic.
                        4- Also do add the background information if necessary to better understand the topic.
                        5- Strictly talk about the topic being asked for.
                        Information Collected:
                        {information}
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
                    You are given a topic and information that is collected for that topic. This topic is a part of a larger lecture being delivered.
                    Instructions:
                    1- Make sure that the part you are generating looks like the continuation of the larger lecture and not feel like it is being taught separately.
                    2- Include coding examples with explanation for them, keep the important information.Dont make the content very small make it large.
                    3- Divide the information into smaller parts.For each smaller part return one add one json object.
                    3- The output should contain the following fields: title, content, and code.
                    4- If any of these fields are absent, return an empty string for that field (e.g., title: "", content: "", code: "").
                    5- Provide one JSON object for each slide.Dont add the json indicator.
                    6- Strictly only add the information relevant to the topic.Like if topic is about introudction only add introduction dont add other stuff like conclusion or something.
                    7- Output should strictly be in json format no extra information just one list of json objects should be returned.
                    Following is the topic and information collected on it:
                    topic: {topic}
                    Information Collected: {information}
                    Expected Output (in JSON format):
                    [
                        {{
                            "title": "Introduction to Pointers",
                            "content": "Pointers are a core concept in C++ that allow for direct memory manipulation, crucial for efficient programming. They are essential for dynamic memory allocation and efficient data structure management, such as linked lists and trees. A pointer holds the memory address of a variable, enabling flexible programming techniques like passing large data structures to functions without copying them.",
                            "code": ""
                        }},
                        {{
                            "title": "",
                            "content": "In C++, declare a pointer with the asterisk (*) symbol. Example:",
                            "code": "int *p;  // Declares a pointer to an integer\nint a = 10;\nint *p = &a;  // p now holds the address of variable a"
                        }},
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
                Your a computer science teacher and specialist at teaching your subject.You have collected information and created slides. Now you being a teacher have to teach them. For that you will need a script which will be divied into sections according to the slides.
                Instructions:
                1-The lecture should not be reading the slides but actually teach what is being said in the slides.
                2- As for the coding examples you should explain what the code is doing not reading the code.
                3- You have to explain what is being said in the slides by adding examples from real world or giving scenarios or by just explaining what is written in slides and what does it mean.
                topic: {topic}
                slides={slides}
            '''
            prompt=ChatPromptTemplate.from_template(template)

            message=prompt.invoke({'topic':topic,'slides':slides})

            lecture_content = self.llm.invoke(message)
            lecture_content = lecture_content.content.strip()
            state['lecture'].append(lecture_content)
        return state


