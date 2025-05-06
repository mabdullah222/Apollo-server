from workflows.PresentationWorkflow import PresentationFlow
import json

if __name__ == "__main__":
    app = PresentationFlow().app
    output = app.invoke({
        "topic": "Teach me Machine Learning",
        "toc": [],
        "resources": [],
        "documents": [],
        "vector_db": "",
        "content": {},
        "slides": [],
        "lecture": [],
        "video_paths":[]
    })
    with open(f'outputs/{output['topic']}.json','w') as ofile:
        json.dump(output,ofile)
