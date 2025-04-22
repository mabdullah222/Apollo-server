from workflows.PresentationWorkflow import PresentationFlow
import json

if __name__ == "__main__":
    app = PresentationFlow().app
    output = app.invoke({
        "topic": "",
        "toc": [],
        "resources": {},
        "documents": {},
        "vector_db": {},
        "content": {},
        "slides": [],
        "lecture": [],
    })
    with open(f'outputs/{output['topic']}.json','w') as ofile:
        del output['vector_db']
        json.dump(output,ofile)
