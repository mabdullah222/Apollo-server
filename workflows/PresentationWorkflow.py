
from langgraph.graph import StateGraph,END
from nodes.PresentationNodes import PresentationState
from nodes.PresentationNodes import Nodes

class PresentationFlow:
    def __init__(self):
        workflow = StateGraph(PresentationState)
        nodes = Nodes()

        workflow.add_node("Topic", nodes.Topic)
        workflow.add_node("SubjectSpecialist", nodes.SubjectSpecialist)
        workflow.add_node("SearchResources", nodes.SearchResources)
        workflow.add_node("ScrapeContent", nodes.ScrapeContent)
        workflow.add_node("StoreInVectorDB", nodes.StoreInVectorDB)
        workflow.add_node("ResearchSpecialist", nodes.ResearchSpecialist)
        workflow.add_node("SlidesMaker", nodes.SlidesMaker)
        workflow.add_node("LectureAgent", nodes.LectureAgent)

        workflow.set_entry_point('Topic')
        workflow.add_edge("Topic", "SubjectSpecialist")
        workflow.add_edge("SubjectSpecialist", "SearchResources")
        workflow.add_edge("SearchResources", "ScrapeContent")
        workflow.add_edge("ScrapeContent", "StoreInVectorDB")
        workflow.add_edge("StoreInVectorDB", "ResearchSpecialist")
        workflow.add_edge("ResearchSpecialist", "SlidesMaker")
        workflow.add_edge("SlidesMaker", "LectureAgent")
        workflow.add_edge("LectureAgent", END)

        self.app = workflow.compile()

        