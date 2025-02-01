
from langgraph.graph import StateGraph,END
from nodes.PresentationNodes import PresentationState
from nodes.PresentationNodes import Nodes

class PresentationFlow:
    def __init__(self):
        nodes = Nodes()
        workflow = StateGraph(PresentationState)
        workflow.add_node('node_topic', nodes.Topic)
        workflow.add_node('node_planner', nodes.SubjectSpecialist)
        workflow.add_node('node_researcher', nodes.ResearchSpecialist)
        workflow.add_node('node_writer', nodes.ContentWriter)
        workflow.add_node('node_slides_maker', nodes.SlidesMaker)
        workflow.add_node('node_lecture',nodes.LectureAgent)

        workflow.set_entry_point('node_topic')

        workflow.add_edge('node_topic', 'node_planner')
        workflow.add_edge('node_planner', 'node_researcher')

        workflow.add_edge('node_researcher', 'node_writer')
        workflow.add_edge('node_writer', 'node_slides_maker')
        workflow.add_edge("node_slides_maker","node_lecture")
        self.app = workflow.compile()

