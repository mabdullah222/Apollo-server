from workflows.PresentationWorkflow import PresentationFlow
import json

if __name__ == "__main__":
    app = PresentationFlow().app
    output = app.invoke({
        "topic": "",
        "undone": [],
        "done": [],
        "information": [],
        "content": [],
        "slides": [],
        'lecture':[]
    })

    with open(f'outputs/{output['topic']}.json','w') as ofile:
        json.dump(output,ofile)