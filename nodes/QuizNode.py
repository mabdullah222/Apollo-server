from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Dict, List
import os
import json

from nodes.PresentationNodes import PresentationState

load_dotenv()

llm = ChatGroq(api_key=os.environ['GROQ_API_KEY_6'], model='llama-3.3-70b-versatile')

def generate_quiz_from_slides_and_lecture(state: PresentationState) -> PresentationState:
    slides = state["slides"]
    lecture_scripts = state["lecture"]
    
    try:
        combined_text = ""
        for slide, script in zip(slides, lecture_scripts):
            combined_text += f"Slide Title: {slide['title']}\nSlide Content: {slide['content']}\nLecture Script: {script}\n\n"

        prompt_template = ChatPromptTemplate.from_template("""
        Based on the following lecture content, generate a quiz of 10 high-quality multiple-choice questions.
        Each question must have four options and one correct answer marked clearly.

        Format:
        [
            {{
                "question": "Your question here?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Option A"
            }},
            ...
        ]

        Lecture Content:
        {content}

        Return only a valid JSON array with no extra commentary.
        """)

        prompt = prompt_template.invoke({"content": combined_text})
        response = llm.invoke(prompt)
        quiz = json.loads(response.content.strip())
        state["quiz"] = quiz  # Optional: you can store in state, or just return it separately

        print("Quiz generation complete.")
    except Exception as e:
        print(f"Quiz generation failed: {e}")
        state["quiz"] = []

    return state
